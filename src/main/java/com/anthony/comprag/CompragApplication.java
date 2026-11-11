package com.anthony.comprag;

import com.anthony.comprag.ingestion.CorpusIngestService;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 金融合规知识库问答系统（RAG）。
 * Day1：语料入库管道；后续：检索问答 API、pgvector、评测体系。
 */
@SpringBootApplication
public class CompragApplication implements ApplicationRunner {

    private final CorpusIngestService corpusIngestService;

    public CompragApplication(CorpusIngestService corpusIngestService) {
        this.corpusIngestService = corpusIngestService;
    }

    public static void main(String[] args) {
        SpringApplication.run(CompragApplication.class, args);
    }

    @Override
    public void run(ApplicationArguments args) {
        corpusIngestService.ingestCorpus();
    }
}
