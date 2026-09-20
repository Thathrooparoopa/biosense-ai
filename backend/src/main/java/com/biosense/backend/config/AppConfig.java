package com.biosense.backend.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
@EnableConfigurationProperties(AiServiceConfig.class)
public class AppConfig {

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }

    @Bean
    public RestClient aiServiceRestClient(
            RestClient.Builder builder,
            AiServiceConfig aiServiceConfig
    ) {
        return builder
                .baseUrl(aiServiceConfig.url())
                .build();
    }
}