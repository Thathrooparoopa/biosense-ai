package com.biosense.backend.dto;

import java.time.LocalDateTime;
import java.util.Map;

public record PredictionHistoryResponse(
        Long id,
        String modelVersion,
        String predictedClass,
        Integer encodedClass,
        Double confidence,
        Map<String, Double> classProbabilities,
        Integer featureCount,
        LocalDateTime createdAt
) {
}