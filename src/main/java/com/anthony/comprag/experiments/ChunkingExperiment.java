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
 * 第 5 周 Day 1：切片策略对比实验（受控实验，不依赖 Spring）。
 * 同语料 + 同嵌入模型 + 同探针集，唯一变量是切片策略。
 * 讲义：importTantlearing/weeks/week05-切片与检索优化/Day1-讲义-切片策略对比实验.md
 */
public class ChunkingExperiment {

    private static final String CORPUS_DIR = "corpus";
    private static final String FILE_EXTENSION = ".txt";
    private static final String METADATA_FILE_NAME = "file_name";
    private static final int TOP_K = 3;

    private static final int BASE_SIZE = 300;
    private static final int BASE_OVERLAP = 30;
    private static final int SMALL_SIZE = 150;
    private static final int SMALL_OVERLAP = 20;
    private static final int LARGE_SIZE = 600;
    private static final int LARGE_OVERLAP = 50;

    /** 探针集：问题 + 期望命中的文件前缀（第 8 周评测集的第一批种子） */
    record Probe(String question, String expectedFilePrefix) {}

    private static final List<Probe> PROBES = List.of(
            new Probe("贷款前要做哪些调查", "01"),
            new Probe("卖理财前要做风险评估吗", "02"),
            new Probe("大额交易报告的标准是什么", "03"),
            new Probe("存款利率要公示吗", "04"),
            new Probe("信用卡催收有什么规范", "05"),
            new Probe("卖保险要告知犹豫期吗", "06"),
            new Probe("客户投诉谁负责", "07"),
            new Probe("涉诈账户怎么管控", "08"),
            new Probe("贷款五级分类标准", "09")
    );

    record Strategy(String name, DocumentSplitter splitter) {}

    record StrategyResult(String name, int segments, int top1Hits, int top3Hits, double avgTop1Score) {}

    public static void main(String[] args) {
        EmbeddingModel model = new BgeSmallZhV15EmbeddingModel();
        List<Path> files = listCorpusFiles();
        System.out.println("语料: " + files.size() + " 个文件 | 探针: " + PROBES.size() + " 条\n");

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

        System.out.println("\n================ 汇总对比 ================");
        System.out.printf("%-18s %6s %10s %10s %8s%n", "策略", "切片数", "Top1命中", "前三命中", "平均Top1分");
        for (StrategyResult r : results) {
            System.out.printf("%-18s %6d %7d/%d %7d/%d %10.4f%n",
                    r.name(), r.segments(), r.top1Hits(), PROBES.size(),
                    r.top3Hits(), PROBES.size(), r.avgTop1Score());
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

        int top1Hits = 0;
        int top3Hits = 0;
        double scoreSum = 0;
        for (Probe probe : PROBES) {
            List<EmbeddingMatch<TextSegment>> matches = search(store, model, probe.question());
            String top1File = sourceFile(matches.get(0));
            boolean top1 = top1File.startsWith(probe.expectedFilePrefix());
            boolean top3 = matches.stream()
                    .anyMatch(m -> sourceFile(m).startsWith(probe.expectedFilePrefix()));
            top1Hits += top1 ? 1 : 0;
            top3Hits += top3 ? 1 : 0;
            scoreSum += matches.get(0).score();
            System.out.printf("  %s %s → top1: %s (%.4f) %s | 进前三: %s%n",
                    top1 ? "✓" : "✗", probe.question(), top1File,
                    matches.get(0).score(), top1 ? "" : "  ←期望:" + probe.expectedFilePrefix() + "-xx", top3 ? "✓" : "✗");
        }
        System.out.printf("  小计: Top1 %d/%d, 前三 %d/%d, 平均分 %.4f%n%n",
                top1Hits, PROBES.size(), top3Hits, PROBES.size(), scoreSum / PROBES.size());
        return new StrategyResult(strategy.name(), segmentCount, top1Hits, top3Hits, scoreSum / PROBES.size());
    }

    private static List<EmbeddingMatch<TextSegment>> search(InMemoryEmbeddingStore<TextSegment> store,
                                                            EmbeddingModel model, String query) {
        Embedding queryEmbedding = model.embed(query).content();
        return store.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(TOP_K)
                        .build())
                .matches();
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
