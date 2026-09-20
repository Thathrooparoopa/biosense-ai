package com.biosense.backend.model;

import java.time.LocalDateTime;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "predictions")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "model_version", nullable = false)
    private String modelVersion;

    @Column(name = "predicted_class", nullable = false)
    private String predictedClass;

    @Column(name = "encoded_class", nullable = false)
    private Integer encodedClass;

    @Column(nullable = false)
    private Double confidence;

    @Column(name = "class_probabilities", nullable = false, columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String classProbabilities;

    @Column(name = "feature_count", nullable = false)
    private Integer featureCount;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    public Prediction() {
    }

    public Prediction(
            User user,
            String modelVersion,
            String predictedClass,
            Integer encodedClass,
            Double confidence,
            String classProbabilities,
            Integer featureCount,
            LocalDateTime createdAt
    ) {
        this.user = user;
        this.modelVersion = modelVersion;
        this.predictedClass = predictedClass;
        this.encodedClass = encodedClass;
        this.confidence = confidence;
        this.classProbabilities = classProbabilities;
        this.featureCount = featureCount;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public Integer getEncodedClass() {
        return encodedClass;
    }

    public void setEncodedClass(Integer encodedClass) {
        this.encodedClass = encodedClass;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getClassProbabilities() {
        return classProbabilities;
    }

    public void setClassProbabilities(String classProbabilities) {
        this.classProbabilities = classProbabilities;
    }

    public Integer getFeatureCount() {
        return featureCount;
    }

    public void setFeatureCount(Integer featureCount) {
        this.featureCount = featureCount;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}