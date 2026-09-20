package com.biosense.backend.service;

import java.util.List;

import org.springframework.stereotype.Service;

import com.biosense.backend.dto.PredictionHistoryResponse;
import com.biosense.backend.model.Prediction;
import com.biosense.backend.model.User;
import com.biosense.backend.repository.PredictionRepository;
import com.biosense.backend.repository.UserRepository;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

@Service
public class PredictionHistoryService {

    private final PredictionRepository predictionRepository;
    private final UserRepository userRepository;
    private final ObjectMapper objectMapper;

    public PredictionHistoryService(
            PredictionRepository predictionRepository,
            UserRepository userRepository,
            ObjectMapper objectMapper
    ) {
        this.predictionRepository = predictionRepository;
        this.userRepository = userRepository;
        this.objectMapper = objectMapper;
    }

    public List<PredictionHistoryResponse> getHistory(String userEmail) {

        User user = userRepository.findByEmail(userEmail)
                .orElseThrow(() ->
                        new IllegalArgumentException(
                                "Authenticated user not found."
                        )
                );

        return predictionRepository
                .findByUserOrderByCreatedAtDesc(user)
                .stream()
                .map(this::toResponse)
                .toList();
    }

    private PredictionHistoryResponse toResponse(Prediction prediction) {

        java.util.Map<String, Double> classProbabilities;

        try {
            classProbabilities = objectMapper.readValue(
                    prediction.getClassProbabilities(),
                    java.util.Map.class
            );
        } catch (JacksonException exception) {
            throw new IllegalStateException(
                    "Unable to read stored class probabilities.",
                    exception
            );
        }

        return new PredictionHistoryResponse(
                prediction.getId(),
                prediction.getModelVersion(),
                prediction.getPredictedClass(),
                prediction.getEncodedClass(),
                prediction.getConfidence(),
                classProbabilities,
                prediction.getFeatureCount(),
                prediction.getCreatedAt()
        );
    }
}