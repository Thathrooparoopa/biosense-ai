package com.biosense.backend.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.biosense.backend.dto.PredictionRequest;
import com.biosense.backend.dto.PredictionResponse;
import com.biosense.backend.model.Prediction;
import com.biosense.backend.service.AiPredictionService;
import com.biosense.backend.service.PredictionPersistenceService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final AiPredictionService aiPredictionService;
    private final PredictionPersistenceService predictionPersistenceService;

    public PredictionController(
            AiPredictionService aiPredictionService,
            PredictionPersistenceService predictionPersistenceService
    ) {
        this.aiPredictionService = aiPredictionService;
        this.predictionPersistenceService = predictionPersistenceService;
    }

    @PostMapping
    public ResponseEntity<PredictionResponse> predict(
            @Valid @RequestBody PredictionRequest request
    ) {
        Authentication authentication =
                SecurityContextHolder.getContext().getAuthentication();

        if (authentication == null
                || !authentication.isAuthenticated()
                || authentication.getName() == null) {

            throw new IllegalStateException(
                    "Authenticated user information is unavailable."
            );
        }

        String userEmail = authentication.getName();

        PredictionResponse response =
                aiPredictionService.predict(request);

        Prediction savedPrediction =
                predictionPersistenceService.save(
                        userEmail,
                        response
                );

        System.out.println("===== PREDICTION PERSISTED =====");
        System.out.println(
                "PREDICTION ID: " + savedPrediction.getId()
        );
        System.out.println(
                "USER EMAIL: " + userEmail
        );

        return ResponseEntity.ok(response);
    }
}