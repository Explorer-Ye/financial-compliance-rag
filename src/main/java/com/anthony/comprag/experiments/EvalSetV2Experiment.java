package com.anthony.comprag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15.BgeSmallZhV15EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * 第 5 周 Day 2：评测集 v2——同义改述（脱开主题词）+ 段级断言 + 负面探针（测误放行）。
 * 讲义：importTantlearing/weeks/week05-切片与检索优化/Day2-讲义-评测集v2.md
 */
public class EvalSetV2Experiment {

    private static final String CORPUS_DIR = "corpus";
    private static final String FILE_EXTENSION = ".txt";
    private static final String METADATA_FILE_NAME = "file_name";
    private static final int TOP_K = 3;
    /** 与生产 RetrievalService 一致的闸门（BGE 实测校准值） */
    private static final double MIN_SCORE_GATE = 0.80;

    private static final int BASE_SIZE = 300;
    private static final int BASE_OVERLAP = 30;
    private static final int SMALL_SIZE = 150;
    private static final int SMALL_OVERLAP = 20;
    private static final int LARGE_SIZE = 600;
    private static final int LARGE_OVERLAP = 50;

    /** 正面探针 v2：改述脱开主题词 + 段级关键词断言 */
    record Probe(String question, String expectedFilePrefix, List<String> expectedKeywords) {}

    private static final List<Probe> POSITIVE_PROBES = List.of(
            new Probe("放贷之前银行要去实地看看吗", "01", List.of("实地调查", "贷前调查")),
            new Probe("给客户卖理财之前要先给他做测评吗", "02", List.of("风险承受能力", "测评")),
            new Probe("一天存5万现金银行要上报吗", "03", List.of("大额交易", "5万")),
            new Probe("拉存款送东西违规吗", "04", List.of("揽储", "变相")),
            new Probe("欠款人不还钱银行能骚扰他家人吗", "05", List.of("催收", "第三人")),
            new Probe("客户买完保险几天内能无理由退吗", "06", List.of("犹豫期")),
            new Probe("客户投诉了找不到人管怎么办", "07", List.of("首问负责")),
            new Probe("把银行卡借给别人用会怎么样", "08", List.of("出借", "出租")),
            new Probe("贷款收不回来在账上怎么记", "09", List.of("五级分类", "不良"))
    );

    /** 负面探针：库里没有，期望被 0.80 闸门拦下（有命中 = 误放行） */
    private static final List<String> NEGATIVE_PROBES = List.of(
            "信用卡积分怎么兑换礼品",
            "银行办公楼装修标准是什么",
            "员工差旅报销流程怎么走"
    );

    record Strategy(String name, DocumentSplitter splitter) {}

    record StrategyResult(String name, int segments, int fileHits, int segmentHits,
                          int falseAccepts) {}

    public static void main(String[] args) {
        EmbeddingModel model = new BgeSmallZhV15EmbeddingModel();
        List<Path> files = listCorpusFiles();
        System.out.println("语料: " + files.size() + " 文件 | 正面探针: " + POSITIVE_PROBES.size()
                + " | 负面探针: " + NEGATIVE_PROBES.size() + " | 闸门: " + MIN_SCORE_GATE + "\n");

        List<Strategy> strategies = List.of(
                new Strategy("递归300/30(基线)", DocumentSplitters.recursive(BASE_SIZE, BASE_OVERLAP)),
                new Strategy("递归150/20(小片)", DocumentSplitters.recursive(SMALL_SIZE, SMALL_OVERLAP)),
                new Strategy("递归600/50(大片)", DocumentSplitters.recursive(LARGE_SIZE, LARGE_OVERLAP)),
                new Strategy("段落300/30", new DocumentByParagraphSplitter(BASE_SIZE, BASE_OVERLAP))
        );

        List<StrategyResult> results = new ArrayList<>();
        for (Strategy strategy : strategies) {
            results.add(runStrategy(strategy, model, files));
        }

        System.out.println("\n================ v2 汇总对比 ================");
        System.out.printf("%-18s %6s %10s %10s %10s%n",
                "策略", "切片数", "文件命中", "段级命中", "误放行");
        for (StrategyResult r : results) {
            System.out.printf("%-18s %6d %7d/%d %7d/%d %7d/%d%n",
                    r.name(), r.segments(),
                    r.fileHits(), POSITIVE_PROBES.size(),
                    r.segmentHits(), POSITIVE_PROBES.size(),
                    r.falseAccepts(), NEGATIVE_PROBES.size());
        }
    }

    private static StrategyResult runStrategy(Strategy strategy, EmbeddingModel model, List<Path> files) {
        System.out.println("=== 策略: " + strategy.name() + " ===");
        InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        int segmentCount = 0;
        for (Path file : files) {
            try {
                Document doc = Document.from(Files.readString(file),
                        new Metadata().put(METADATA_FILE_NAME, file.getFileName().toString()));
                List<TextSegment> segments = strategy.splitter().split(doc);
                List<Embedding> embeddings = model.embedAll(segments).content();
                store.addAll(embeddings, segments);
                segmentCount += segments.size();
            } catch (IOException e) {
                throw new IllegalStateException("读取语料失败: " + file, e);
            }
        }

        int fileHits = 0;
        int segmentHits = 0;
        for (Probe probe : POSITIVE_PROBES) {
            List<EmbeddingMatch<TextSegment>> matches = search(store, model, probe.question(), TOP_K);
            EmbeddingMatch<TextSegment> top1 = matches.get(0);
            boolean fileHit = sourceFile(top1).startsWith(probe.expectedFilePrefix());
            boolean segmentHit = fileHit && containsAny(top1.embedded().text(), probe.expectedKeywords());
            fileHits += fileHit ? 1 : 0;
            segmentHits += segmentHit ? 1 : 0;
            System.out.printf("  %s%s %s → %s (%.4f) 段级%s%n",
                    fileHit ? "✓" : "✗", segmentHit ? "✓" : "✗",
                    probe.question(), sourceFile(top1), top1.score(),
                    segmentHit ? "✓" : "✗" + (fileHit ? "(文件对段落错)" : ""));
        }

        int falseAccepts = 0;
        for (String negative : NEGATIVE_PROBES) {
            List<EmbeddingMatch<TextSegment>> gated = search(store, model, negative, 1);
            boolean leaked = !gated.isEmpty();
            falseAccepts += leaked ? 1 : 0;
            System.out.printf("  %s [负面] %s → %s%n",
                    leaked ? "⚠误放行" : "✓已拦截", negative,
                    leaked ? sourceFile(gated.get(0)) + " (" + String.format("%.4f", gated.get(0).score()) + ")" : "闸门以下");
        }
        System.out.printf("  小计: 文件 %d/%d, 段级 %d/%d, 误放行 %d/%d%n%n",
                fileHits, POSITIVE_PROBES.size(), segmentHits, POSITIVE_PROBES.size(),
                falseAccepts, NEGATIVE_PROBES.size());
        return new StrategyResult(strategy.name(), segmentCount, fileHits, segmentHits, falseAccepts);
    }

    /** 带闸门的检索：minScore 以下不返回（与生产行为一致） */
    private static List<EmbeddingMatch<TextSegment>> search(InMemoryEmbeddingStore<TextSegment> store,
                                                            EmbeddingModel model, String query, int topK) {
        Embedding queryEmbedding = model.embed(query).content();
        return store.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(topK)
                        .minScore(MIN_SCORE_GATE)
                        .build())
                .matches();
    }

    private static boolean containsAny(String text, List<String> keywords) {
        return keywords.stream().anyMatch(text::contains);
    }

    private static String sourceFile(EmbeddingMatch<TextSegment> match) {
        String file = match.embedded().metadata().getString(METADATA_FILE_NAME);
        return file != null ? file : "(未知来源)";
    }

    private static List<Path> listCorpusFiles() {
        Path dir = Path.of(CORPUS_DIR);
        if (!Files.isDirectory(dir)) {
            throw new IllegalStateException("语料目录不存在: " + dir.toAbsolutePath()
                    + "（请在项目根目录运行）");
        }
        try (Stream<Path> paths = Files.list(dir)) {
            List<Path> files = new ArrayList<>(paths
                    .filter(p -> p.getFileName().toString().endsWith(FILE_EXTENSION))
                    .toList());
            files.sort(Path::compareTo);
            return files;
        } catch (IOException e) {
            throw new IllegalStateException("扫描语料目录失败: " + dir, e);
        }
    }
}
