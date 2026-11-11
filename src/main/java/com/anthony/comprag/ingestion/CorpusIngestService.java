package com.anthony.comprag.ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * 语料入库管道：扫描 corpus 目录 → 逐文件解析 → 切片 → 向量化 → 入库。
 * 切片参数集中为常量，第 5 周做切片策略对比实验时只动这里。
 */
@Service
public class CorpusIngestService {

    private static final int SEGMENT_MAX_CHARS = 300;
    private static final int SEGMENT_OVERLAP_CHARS = 30;
    private static final String CORPUS_FILE_EXTENSION = ".txt";
    private static final String METADATA_FILE_NAME = "file_name";

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final Path corpusDir;

    public CorpusIngestService(EmbeddingModel embeddingModel,
                               EmbeddingStore<TextSegment> embeddingStore,
                               @Value("${corpus.dir}") String corpusDir) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
        this.corpusDir = Path.of(corpusDir);
    }

    public void ingestCorpus() {
        List<Path> files = listCorpusFiles();
        System.out.println(">>> 语料目录: " + corpusDir.toAbsolutePath() + "，共 " + files.size() + " 个文件");
        int totalSegments = 0;
        for (Path file : files) {
            int segments = ingestOneFile(file);
            totalSegments += segments;
            System.out.println("    - " + file.getFileName() + " → " + segments + " 个切片");
        }
        System.out.println(">>> 入库完成，共 " + totalSegments + " 个切片");
    }

    private int ingestOneFile(Path file) {
        Document document = parse(file);
        List<TextSegment> segments = DocumentSplitters
                .recursive(SEGMENT_MAX_CHARS, SEGMENT_OVERLAP_CHARS)
                .split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        embeddingStore.addAll(embeddings, segments);
        return segments.size();
    }

    private Document parse(Path file) {
        try {
            String text = Files.readString(file);
            // 文件名写入元数据——切片器会把它传播到每个切片，检索时用它做引用溯源
            Metadata metadata = new Metadata().put(METADATA_FILE_NAME, file.getFileName().toString());
            return Document.from(text, metadata);
        } catch (IOException e) {
            throw new IllegalStateException("语料解析失败: " + file, e);
        }
    }

    private List<Path> listCorpusFiles() {
        if (!Files.isDirectory(corpusDir)) {
            throw new IllegalStateException("语料目录不存在: " + corpusDir.toAbsolutePath());
        }
        try (Stream<Path> paths = Files.list(corpusDir)) {
            List<Path> files = new ArrayList<>(paths
                    .filter(p -> p.getFileName().toString().endsWith(CORPUS_FILE_EXTENSION))
                    .toList());
            files.sort(Path::compareTo);
            return files;
        } catch (IOException e) {
            throw new IllegalStateException("扫描语料目录失败: " + corpusDir, e);
        }
    }
}
