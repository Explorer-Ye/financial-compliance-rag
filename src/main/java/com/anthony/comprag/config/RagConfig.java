package com.anthony.comprag.config;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15.BgeSmallZhV15EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 基础组件（可替换部件集中地）。
 * 嵌入模型由 app.embedding.provider 配置驱动：local=本地BGE（离线免费），api=智谱 embedding-3。
 * 注意：换嵌入模型后，RetrievalService 的 minScore 闸门必须用实测分数重新校准（两组向量空间不可比）。
 * 路线：第6周向量库换 pgvector——同样只改这里。
 */
@Configuration
public class RagConfig {

    private static final String PROVIDER_LOCAL = "local";
    private static final String PROVIDER_API = "api";
    private static final String ZHIPU_API_KEY_ENV = "ZHIPU_API_KEY";

    private static final String CHAT_BASE_URL = "https://api.deepseek.com";
    private static final String CHAT_MODEL_NAME = "deepseek-chat";
    private static final String DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY";
    private static final double TEMP_DETERMINISTIC = 0.0;

    @Bean
    public EmbeddingModel embeddingModel(
            @Value("${app.embedding.provider}") String provider,
            @Value("${app.embedding.api.base-url}") String apiBaseUrl,
            @Value("${app.embedding.api.model}") String apiModel) {
        if (PROVIDER_LOCAL.equals(provider)) {
            return new BgeSmallZhV15EmbeddingModel();
        }
        if (PROVIDER_API.equals(provider)) {
            String apiKey = System.getenv(ZHIPU_API_KEY_ENV);
            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException("缺少环境变量 " + ZHIPU_API_KEY_ENV
                        + "（智谱平台创建），请在运行配置中设置后重启");
            }
            // 智谱 /embeddings 接口兼容 OpenAI 协议，所以复用 OpenAiEmbeddingModel 换 baseUrl
            return OpenAiEmbeddingModel.builder()
                    .baseUrl(apiBaseUrl)
                    .apiKey(apiKey)
                    .modelName(apiModel)
                    .build();
        }
        throw new IllegalStateException("未知的嵌入提供方: " + provider + "（可选: local / api）");
    }

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

    @Bean
    public ChatModel chatModel() {
        String apiKey = System.getenv(DEEPSEEK_API_KEY_ENV);
        if (apiKey == null || apiKey.isBlank()) {
            // 快速失败：启动即报清楚缺什么，好过运行时每个请求都 500
            throw new IllegalStateException("缺少环境变量 " + DEEPSEEK_API_KEY_ENV
                    + "，请在运行配置（Edit Configurations → Environment variables）中设置后重启");
        }
        return OpenAiChatModel.builder()
                .baseUrl(CHAT_BASE_URL)
                .apiKey(apiKey)
                .modelName(CHAT_MODEL_NAME)
                .temperature(TEMP_DETERMINISTIC)
                .build();
    }
}
