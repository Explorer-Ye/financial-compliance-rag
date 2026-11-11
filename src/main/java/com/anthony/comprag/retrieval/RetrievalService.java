package com.anthony.comprag.retrieval;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 检索层：问题向量化 → Top-K 检索（带相关度闸门）。
 * 独立成层的原因：检索质量决定 RAG 上限，且必须能脱离 LLM 单独测试与调优。
 */
@Service
public class RetrievalService {

    private static final int TOP_K = 4;
    /**
     * 相关度闸门：0.80（基于 BGE-small-zh 在当前语料上的实测分布校准）。
     * 校准数据(2026-11-11)：库内问题 top1 = 0.871~0.876，库外问题 top1 = 0.769。
     * 注意：该阈值与嵌入模型强绑定，第5周换 API 嵌入后必须用同样方法重新校准（第8周做成正式评测集）。
     */
    private static final double MIN_SCORE_GATE = 0.80;
    private static final String METADATA_FILE_NAME = "file_name";
    private static final String UNKNOWN_SOURCE = "(未知来源)";

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public RetrievalService(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> embeddingStore) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
    }

    /** 一次检索命中：来源文件 + 相关度 + 原文 */
    public record RetrievalHit(String file, double score, String text) {}

    public List<RetrievalHit> search(String query) {
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        List<EmbeddingMatch<TextSegment>> matches = embeddingStore
                .search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(TOP_K)
                        .minScore(MIN_SCORE_GATE)
                        .build())
                .matches();
        return matches.stream()
                .map(m -> new RetrievalHit(
                        sourceFile(m.embedded()),
                        m.score(),
                        m.embedded().text()))
                .toList();
    }

    private String sourceFile(TextSegment segment) {
        String file = segment.metadata().getString(METADATA_FILE_NAME);
        return file != null ? file : UNKNOWN_SOURCE;
    }
}
