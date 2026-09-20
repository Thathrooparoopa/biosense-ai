package com.biosense.backend.dto;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PredictionResponse(

        String status,

        @JsonProperty("model_version")
        String modelVersion,

        Prediction prediction,

        @JsonProperty("class_probabilities")
        Map<String, Double> classProbabilities,

        @JsonProperty("feature_count")
        Integer featureCount,

        String message,

        String disclaimer

) {

    public record Prediction(

            @JsonProperty("class_code")
            String classCode,

            @JsonProperty("encoded_class")
            Integer encodedClass,

            Double confidence

    ) {
    }
}