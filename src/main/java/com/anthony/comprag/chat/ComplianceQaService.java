package com.anthony.comprag.chat;

import com.anthony.comprag.retrieval.RetrievalService;
import com.anthony.comprag.retrieval.RetrievalService.RetrievalHit;
import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 问答层：检索命中 → 拼 Prompt（仅依据资料作答）→ 生成 → 返回答案+引用。
 * 检索零命中时直接拒答，不调用 LLM（省一次请求，且从机制上杜绝"无依据作答"）。
 */
@Service
public class ComplianceQaService {

    private static final String RAG_PROMPT_TEMPLATE =
            "你是银行合规知识助手。仅根据下面的资料回答问题；资料中没有相关内容时"
                    + "明确回答\"根据现有资料无法确定\"，不要编造。\n\n【资料】\n%s\n\n【问题】%s";
    private static final String NO_HIT_ANSWER = "知识库中未检索到相关资料，无法回答该问题。";

    private final RetrievalService retrievalService;
    private final ChatModel chatModel;

    public ComplianceQaService(RetrievalService retrievalService, ChatModel chatModel) {
        this.retrievalService = retrievalService;
        this.chatModel = chatModel;
    }

    public record AskResult(String answer, List<RetrievalHit> citations) {}

    public AskResult ask(String question) {
        List<RetrievalHit> hits = retrievalService.search(question);
        if (hits.isEmpty()) {
            return new AskResult(NO_HIT_ANSWER, List.of());
        }
        StringBuilder context = new StringBuilder();
        for (RetrievalHit hit : hits) {
            context.append("- 【来源: ").append(hit.file()).append("】")
                    .append(hit.text()).append("\n");
        }
        String answer = chatModel.chat(RAG_PROMPT_TEMPLATE.formatted(context, question));
        return new AskResult(answer, hits);
    }
}
