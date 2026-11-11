package com.anthony.comprag.web;

import com.anthony.comprag.chat.ComplianceQaService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ComplianceController {

    private final ComplianceQaService qaService;

    public ComplianceController(ComplianceQaService qaService) {
        this.qaService = qaService;
    }

    public record AskRequest(String question) {}

    @PostMapping("/ask")
    public ComplianceQaService.AskResult ask(@RequestBody AskRequest request) {
        return qaService.ask(request.question());
    }
}
