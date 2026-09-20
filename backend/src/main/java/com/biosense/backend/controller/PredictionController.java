package com.biosense.backend.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.biosense.backend.dto.PredictionRequest;
import com.biosense.backend.dto.PredictionResponse;
import com.biosense.backend.service.AiPredictionService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final AiPredictionService aiPredictionService;

    public PredictionController(
            AiPredictionService aiPredictionService
    ) {
        this.aiPredictionService = aiPredictionService;
    }

    @PostMapping
    public ResponseEntity<PredictionResponse> predict(
            @Valid @RequestBody PredictionRequest request
    ) {
        return ResponseEntity.ok(
                aiPredictionService.predict(request)
        );
    }
}