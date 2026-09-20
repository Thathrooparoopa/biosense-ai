package com.biosense.backend.service;

import java.time.LocalDateTime;

import org.springframework.stereotype.Service;

import com.biosense.backend.dto.PredictionResponse;
import com.biosense.backend.model.Prediction;
import com.biosense.backend.model.User;
import com.biosense.backend.repository.PredictionRepository;
import com.biosense.backend.repository.UserRepository;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

@Service
public class PredictionPersistenceService {

    private final PredictionRepository predictionRepository;
    private final UserRepository userRepository;
    private final ObjectMapper objectMapper;

    public PredictionPersistenceService(
            PredictionRepository predictionRepository,
            UserRepository userRepository,
            ObjectMapper objectMapper
    ) {
        this.predictionRepository = predictionRepository;
        this.userRepository = userRepository;
        this.objectMapper = objectMapper;
    }

    public Prediction save(
            String userEmail,
            PredictionResponse response
    ) {

        User user = userRepository.findByEmail(userEmail)
                .orElseThrow(() ->
                        new IllegalArgumentException(
                                "Authenticated user not found."
                        )
                );

        if (response == null || response.prediction() == null) {
            throw new IllegalArgumentException(
                    "Prediction response is incomplete."
            );
        }

        final String classProbabilitiesJson;

        try {
            classProbabilitiesJson =
                    objectMapper.writeValueAsString(
                            response.classProbabilities()
                    );
        } catch (JacksonException exception) {
            throw new IllegalStateException(
                    "Unable to serialize class probabilities.",
                    exception
            );
        }

        Prediction prediction = new Prediction(
                user,
                response.modelVersion(),
                response.prediction().classCode(),
                response.prediction().encodedClass(),
                response.prediction().confidence(),
                classProbabilitiesJson,
                response.featureCount(),
                LocalDateTime.now()
        );

        return predictionRepository.save(prediction);
    }
}