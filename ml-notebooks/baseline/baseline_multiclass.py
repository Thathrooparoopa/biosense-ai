from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
)

RESULTS_PATH = OUTPUT_DIR / "baseline_multiclass_results.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Utility functions
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
    print("BioSense AI - Phase 12 Multi-Class Baseline")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"Dataset path: {DATASET_PATH}")

    return df


def prepare_features(df):
    """Prepare ML features and encoded target labels."""

    # Patient ID is an identifier, not an ML feature.
    excluded_columns = [
        "patient_id",
        "diagnosis",
    ]

    # Exclude sample_count features from the baseline model.
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


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, labels):
    """Train and evaluate one baseline model."""

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=labels,
        zero_division=0,
    )

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")

    print("\nClassification Report")
    print("-" * 70)
    print(report)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


# ============================================================
# Main
# ============================================================

def main():
    df = load_dataset()

    X, y, label_encoder, feature_columns = prepare_features(df)

    # --------------------------------------------------------
    # Patient-level split
    # --------------------------------------------------------
    #
    # Each row represents one patient-level observation in the
    # processed dataset, so the split is performed at the
    # patient-record level.
    #
    # Stratification keeps all diagnostic groups represented
    # in both training and testing sets.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset split")
    print("-" * 70)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # ========================================================
    # Model 1 - Logistic Regression
    # ========================================================

    logistic_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    logistic_result = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder.classes_,
    )

    # ========================================================
    # Model 2 - Random Forest
    # ========================================================

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest_result = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder.classes_,
    )

    # ========================================================
    # Model 3 - XGBoost
    # ========================================================

    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost_result = evaluate_model(
        "XGBoost",
        xgboost_model,
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder.classes_,
    )

    # ========================================================
    # Save results
    # ========================================================

    results = [
        logistic_result,
        random_forest_result,
        xgboost_result,
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("BioSense AI - Phase 12 Multi-Class Baseline Results\n")
        file.write("=" * 70 + "\n\n")

        file.write("Dataset\n")
        file.write("-" * 70 + "\n")
        file.write(f"Dataset shape: {df.shape}\n")
        file.write(f"ML feature count: {len(feature_columns)}\n")
        file.write(f"Number of classes: {len(label_encoder.classes_)}\n")
        file.write(f"Training samples: {len(X_train)}\n")
        file.write(f"Testing samples: {len(X_test)}\n")
        file.write(f"Random state: {RANDOM_STATE}\n")
        file.write(f"Test size: {TEST_SIZE}\n\n")

        file.write("Class Mapping\n")
        file.write("-" * 70 + "\n")

        for encoded_value, class_name in enumerate(
            label_encoder.classes_
        ):
            file.write(
                f"{encoded_value} -> {class_name}\n"
            )

        file.write("\n")

        for result in results:

            file.write("=" * 70 + "\n")
            file.write(f"{result['model_name']}\n")
            file.write("=" * 70 + "\n")

            file.write(
                f"Accuracy: {result['accuracy']:.4f}\n"
            )

            file.write(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy']:.4f}\n"
            )

            file.write(
                f"Macro F1: {result['macro_f1']:.4f}\n\n"
            )

            file.write(
                result["classification_report"]
            )

            file.write("\n")

    print("\n" + "=" * 70)
    print("PHASE 12 BASELINE COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()