from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "soh_multiclass_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ai-service"
    / "models"
)

MODEL_VERSION = "biosense-v1"

MODEL_PATH = MODEL_DIR / "biosense_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
N_ESTIMATORS = 300


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset():
    """Load the processed multi-class S-O-H feature dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Run the Phase 11 preprocessing script first."
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("BioSense AI - Phase 14 Model Persistence")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"Dataset path: {DATASET_PATH}")

    return df


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(df):
    """
    Prepare the exact feature matrix and encoded target.

    This follows the same feature-selection logic used by
    the Phase 12 baseline script.
    """

    excluded_columns = [
        "patient_id",
        "diagnosis",
    ]

    sample_count_columns = [
        column
        for column in df.columns
        if column.endswith("_sample_count")
    ]

    excluded_columns.extend(sample_count_columns)

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()

    y_raw = df["diagnosis"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("\nFeature preparation")
    print("-" * 70)
    print(f"Total dataset columns: {len(df.columns)}")
    print(f"Excluded sample_count features: {len(sample_count_columns)}")
    print(f"Final ML feature count: {len(feature_columns)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    print("\nClass mapping:")

    for encoded_value, class_name in enumerate(label_encoder.classes_):
        print(f"  {encoded_value} -> {class_name}")

    return X, y, label_encoder, feature_columns


# ============================================================
# Model Training
# ============================================================

def train_model(X, y):
    """Train the Random Forest model used for biosense-v1."""

    print("\n" + "=" * 70)
    print("Training Random Forest")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X, y)

    print("\nRandom Forest training complete.")

    return model


# ============================================================
# Model Persistence
# ============================================================

def save_artifacts(
    model,
    label_encoder,
    feature_columns,
    df,
):
    """Save all artifacts required for reproducible inference."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save label encoder
    # --------------------------------------------------------

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    # --------------------------------------------------------
    # Save exact feature ordering
    # --------------------------------------------------------

    with open(
        FEATURE_COLUMNS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_columns,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save model metadata
    # --------------------------------------------------------

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "RandomForestClassifier",
        "dataset": "S-O-H",
        "task": "multiclass_classification",
        "feature_count": len(feature_columns),
        "classes": label_encoder.classes_.tolist(),
        "training_samples": len(df),
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "excluded_columns": [
            "patient_id",
            "diagnosis",
        ],
        "excluded_feature_pattern": "*_sample_count",
        "research_use_only": True,
        "clinical_diagnostic_use": False,
    }

    with open(
        MODEL_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 70)

    print(f"\nModel version: {MODEL_VERSION}")

    print(f"\nModel:")
    print(MODEL_PATH)

    print(f"\nLabel encoder:")
    print(LABEL_ENCODER_PATH)

    print(f"\nFeature columns:")
    print(FEATURE_COLUMNS_PATH)

    print(f"\nMetadata:")
    print(MODEL_METADATA_PATH)


# ============================================================
# Validation
# ============================================================

def validate_saved_artifacts():
    """Verify that all expected model artifacts exist."""

    print("\n" + "=" * 70)
    print("VALIDATING SAVED ARTIFACTS")
    print("=" * 70)

    artifact_paths = [
        MODEL_PATH,
        LABEL_ENCODER_PATH,
        FEATURE_COLUMNS_PATH,
        MODEL_METADATA_PATH,
    ]

    all_valid = True

    for artifact_path in artifact_paths:
        exists = artifact_path.exists()

        status = "OK" if exists else "MISSING"

        print(f"{status:8} {artifact_path}")

        if not exists:
            all_valid = False

    if not all_valid:
        raise FileNotFoundError(
            "One or more model artifacts were not created."
        )

    print("\nAll Phase 14 model artifacts are present.")


# ============================================================
# Main
# ============================================================

def main():
    df = load_dataset()

    X, y, label_encoder, feature_columns = prepare_features(df)

    model = train_model(
        X,
        y,
    )

    save_artifacts(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        df=df,
    )

    validate_saved_artifacts()

    print("\n" + "=" * 70)
    print("PHASE 14 MODEL PERSISTENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()