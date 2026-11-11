package com.anthony.comprag.config;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15.BgeSmallZhV15EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 基础组件。当前：本地嵌入 + 内存向量库 + DeepSeek 生成。
 * 路线：第5周嵌入升级为 API 级；第6周向量库换 pgvector——都只改这里的 Bean，业务零改动。
 */
@Configuration
public class RagConfig {

    private static final String BASE_URL = "https://api.deepseek.com";
    private static final String MODEL_NAME = "deepseek-chat";
    private static final String API_KEY_ENV = "DEEPSEEK_API_KEY";
    private static final double TEMP_DETERMINISTIC = 0.0;

    @Bean
    public EmbeddingModel embeddingModel() {
        return new BgeSmallZhV15EmbeddingModel();
    }

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

    @Bean
    public ChatModel chatModel() {
        String apiKey = System.getenv(API_KEY_ENV);
        if (apiKey == null || apiKey.isBlank()) {
            // 快速失败：启动即报清楚缺什么，好过运行时每个请求都 500
            throw new IllegalStateException("缺少环境变量 " + API_KEY_ENV
                    + "，请在运行配置（Edit Configurations → Environment variables）中设置后重启");
        }
        return OpenAiChatModel.builder()
                .baseUrl(BASE_URL)
                .apiKey(apiKey)
                .modelName(MODEL_NAME)
                .temperature(TEMP_DETERMINISTIC)
                .build();
    }
}
