package com.biosense.backend.service;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;

import com.biosense.backend.config.AiServiceConfig;
import com.biosense.backend.dto.PredictionRequest;
import com.biosense.backend.dto.PredictionResponse;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

@Service
public class AiPredictionService {

    private final AiServiceConfig aiServiceConfig;
        private final ObjectMapper objectMapper;

    public AiPredictionService(
                        AiServiceConfig aiServiceConfig,
                        ObjectMapper objectMapper
    ) {
        this.aiServiceConfig = aiServiceConfig;
                this.objectMapper = objectMapper;
    }

    public PredictionResponse predict(PredictionRequest request) {

        try {
            Map<String, Object> aiRequest = new LinkedHashMap<>();
            aiRequest.put("sensor_data", request.sensorData());
            String aiRequestJson = objectMapper.writeValueAsString(aiRequest);

            System.out.println("===== AI SERVICE REQUEST =====");
            System.out.println(
                    "AI SERVICE URL: "
                            + aiServiceConfig.url()
            );
            System.out.println(
                    "AI SERVICE PATH: "
                            + aiServiceConfig.predictionPath()
            );
            System.out.println(
                    "FEATURE COUNT: "
                            + request.sensorData().size()
            );

            byte[] requestBytes =
                    aiRequestJson.getBytes(StandardCharsets.UTF_8);
            HttpURLConnection connection =
                    (HttpURLConnection) new URL(
                            aiServiceConfig.url()
                                    + aiServiceConfig.predictionPath()
                    ).openConnection();

            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setFixedLengthStreamingMode(requestBytes.length);
            connection.setRequestProperty(
                    "Content-Type",
                    MediaType.APPLICATION_JSON_VALUE
            );
            connection.setRequestProperty(
                    "Accept",
                    MediaType.APPLICATION_JSON_VALUE
            );

            try (OutputStream outputStream = connection.getOutputStream()) {
                outputStream.write(requestBytes);
            }

            int responseStatus = connection.getResponseCode();
            InputStream responseStream = responseStatus >= 400
                    ? connection.getErrorStream()
                    : connection.getInputStream();
            String responseBody = responseStream == null
                    ? ""
                    : new String(
                            responseStream.readAllBytes(),
                            StandardCharsets.UTF_8
                    );

            if (responseStatus < 200 || responseStatus >= 300) {
                System.err.println(
                        "AI SERVICE HTTP ERROR: "
                                + responseStatus
                );
                System.err.println(
                        "AI SERVICE RESPONSE: "
                                + responseBody
                );
                throw new IllegalStateException(
                        "AI service rejected the prediction request: "
                                + responseBody
                );
            }

            PredictionResponse response =
                    objectMapper.readValue(
                            responseBody,
                            PredictionResponse.class
                    );

            if (response == null) {
                throw new IllegalStateException(
                        "AI service returned an empty response."
                );
            }

            System.out.println("===== AI SERVICE RESPONSE =====");
            System.out.println(
                    "MODEL VERSION: "
                            + response.modelVersion()
            );

            if (response.prediction() != null) {
                System.out.println(
                        "PREDICTION: "
                                + response.prediction().classCode()
                );

                System.out.println(
                        "CONFIDENCE: "
                                + response.prediction().confidence()
                );
            }

            return response;

        } catch (JacksonException exception) {

            throw new IllegalStateException(
                    "Unable to serialize the AI service prediction request.",
                    exception
            );

        } catch (IOException exception) {

            System.err.println(
                    "AI SERVICE CONNECTION ERROR: "
                            + exception.getMessage()
            );

            throw new IllegalStateException(
                    "Unable to reach the BioSense AI service at "
                            + aiServiceConfig.url(),
                    exception
            );

        }
    }
}