package com.biosense.backend.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.biosense.backend.model.Prediction;
import com.biosense.backend.model.User;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    List<Prediction> findByUserOrderByCreatedAtDesc(User user);
}