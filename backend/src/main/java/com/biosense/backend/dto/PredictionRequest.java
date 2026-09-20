package com.biosense.backend.dto;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.validation.constraints.NotEmpty;

public record PredictionRequest(

        @JsonProperty("sensor_data")
        @NotEmpty(message = "sensor_data must not be empty")
        Map<String, Double> sensorData

) {
}