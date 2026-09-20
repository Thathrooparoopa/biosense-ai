from pathlib import Path
import json
from typing import Dict

import joblib
import pandas as pd
from fastapi import HTTPException


class PredictionService:
    """
    Service responsible for BioSense AI model inference.

    The service loads the persisted biosense-v1 model and
    associated artifacts from the ai-service/models directory.
    """

    def __init__(self):
        # ----------------------------------------------------
        # Model artifact paths
        # ----------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_dir = self.project_root / "models"

        self.model_path = (
            self.model_dir / "biosense_model.joblib"
        )

        self.label_encoder_path = (
            self.model_dir / "label_encoder.joblib"
        )

        self.feature_columns_path = (
            self.model_dir / "feature_columns.json"
        )

        self.metadata_path = (
            self.model_dir / "model_metadata.json"
        )

        # ----------------------------------------------------
        # Load persisted artifacts
        # ----------------------------------------------------

        self.model = None
        self.label_encoder = None
        self.feature_columns = []
        self.metadata = {}

        self._load_artifacts()

    # ========================================================
    # Model Loading
    # ========================================================

    def _load_artifacts(self):
        """Load all biosense-v1 model artifacts."""

        required_files = [
            self.model_path,
            self.label_encoder_path,
            self.feature_columns_path,
            self.metadata_path,
        ]

        missing_files = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "BioSense AI model artifacts are missing:\n"
                + "\n".join(missing_files)
            )

        self.model = joblib.load(
            self.model_path
        )

        self.label_encoder = joblib.load(
            self.label_encoder_path
        )

        with open(
            self.feature_columns_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.feature_columns = json.load(file)

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # ----------------------------------------------------
        # Basic artifact consistency validation
        # ----------------------------------------------------

        metadata_feature_count = self.metadata.get(
            "feature_count"
        )

        if metadata_feature_count != len(
            self.feature_columns
        ):
            raise ValueError(
                "Model metadata feature count does not "
                "match feature_columns.json."
            )

        model_version = self.metadata.get(
            "model_version",
            "unknown",
        )

        print("=" * 70)
        print("BioSense AI model loaded successfully")
        print("=" * 70)
        print(f"Model version: {model_version}")
        print(
            f"Feature count: {len(self.feature_columns)}"
        )
        print(
            f"Class count: {len(self.label_encoder.classes_)}"
        )
        print(
            f"Model type: {type(self.model).__name__}"
        )

    # ========================================================
    # Feature Validation
    # ========================================================

    def _validate_features(
        self,
        sensor_data: Dict[str, float],
    ):
        """
        Validate that the request contains exactly the
        feature names required by the persisted model.
        """

        required_features = set(
            self.feature_columns
        )

        received_features = set(
            sensor_data.keys()
        )

        missing_features = sorted(
            required_features - received_features
        )

        unexpected_features = sorted(
            received_features - required_features
        )

        if missing_features or unexpected_features:
            detail = {
                "message": (
                    "Invalid sensor feature input. "
                    "The request must contain exactly "
                    "the features required by the persisted model."
                ),
                "expected_feature_count": len(
                    self.feature_columns
                ),
                "received_feature_count": len(
                    sensor_data
                ),
            }

            if missing_features:
                detail["missing_features"] = (
                    missing_features
                )

            if unexpected_features:
                detail["unexpected_features"] = (
                    unexpected_features
                )

            raise HTTPException(
                status_code=400,
                detail=detail,
            )

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        sensor_data: Dict[str, float],
    ) -> dict:
        """
        Run model inference on the supplied feature values.
        """

        if self.model is None:
            raise RuntimeError(
                "BioSense AI model is not loaded."
            )

        # ----------------------------------------------------
        # Validate incoming features
        # ----------------------------------------------------

        self._validate_features(
            sensor_data
        )

        # ----------------------------------------------------
        # Build DataFrame using the exact feature order
        # saved during Phase 14.
        # ----------------------------------------------------

        feature_row = {
            feature: sensor_data[feature]
            for feature in self.feature_columns
        }

        input_df = pd.DataFrame(
            [feature_row],
            columns=self.feature_columns,
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        predicted_encoded_class = int(
            self.model.predict(input_df)[0]
        )

        predicted_class = (
            self.label_encoder.inverse_transform(
                [predicted_encoded_class]
            )[0]
        )

        # ----------------------------------------------------
        # Class probabilities
        # ----------------------------------------------------

        probabilities = self.model.predict_proba(
            input_df
        )[0]

        class_probabilities = {}

        for encoded_class, probability in zip(
            self.model.classes_,
            probabilities,
        ):
            class_name = (
                self.label_encoder.inverse_transform(
                    [int(encoded_class)]
                )[0]
            )

            class_probabilities[
                str(class_name)
            ] = round(
                float(probability),
                6,
            )

        # ----------------------------------------------------
        # Model confidence
        # ----------------------------------------------------

        confidence = round(
            float(max(probabilities)),
            6,
        )

        # ----------------------------------------------------
        # Research-safe response
        # ----------------------------------------------------

        return {
            "status": "success",
            "model_version": self.metadata.get(
                "model_version",
                "unknown",
            ),
            "prediction": {
                "class_code": str(
                    predicted_class
                ),
                "encoded_class": predicted_encoded_class,
                "confidence": confidence,
            },
            "class_probabilities": class_probabilities,
            "feature_count": len(
                self.feature_columns
            ),
            "message": (
                "Predicted diagnostic-associated "
                "VOC pattern from the research model."
            ),
            "disclaimer": (
                "This application is an experimental "
                "research prototype and is not a medical "
                "diagnostic device. Its results must not "
                "be used to diagnose, treat, or rule out "
                "cancer. Medical decisions should be made "
                "only by qualified healthcare professionals."
            ),
        }