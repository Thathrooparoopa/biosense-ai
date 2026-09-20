package com.biosense.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "biosense.ai-service")
public record AiServiceConfig(
        String url,
        String predictionPath
) {
}