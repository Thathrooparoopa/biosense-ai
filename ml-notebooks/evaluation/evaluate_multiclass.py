from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

PLOTS_DIR = (
    PROJECT_ROOT
    / "screenshots"
    / "ml-evaluation"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "phase13_evaluation_results.txt"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    """Load the processed multi-class dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Run the Phase 11 preprocessing script first."
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("BioSense AI - Phase 13 ML Evaluation")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    return df


def prepare_data(df):
    """Prepare features and encoded labels."""

    excluded_columns = {
        "patient_id",
        "diagnosis",
    }

    sample_count_columns = [
        column
        for column in df.columns
        if column.endswith("_sample_count")
    ]

    excluded_columns.update(sample_count_columns)

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df["diagnosis"].astype(str)
    )

    return (
        X,
        y,
        label_encoder,
        feature_columns,
    )


# ============================================================
# Models
# ============================================================

def create_models(number_of_classes):
    """Create baseline models for evaluation."""

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=number_of_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation(
    models,
    X,
    y,
):
    """Run stratified cross-validation."""

    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    results = {}

    for model_name, model in models.items():

        print(f"\n{model_name}")
        print("-" * 70)

        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        model_result = {}

        for metric_name in scoring:

            scores = cv_result[
                f"test_{metric_name}"
            ]

            mean_score = scores.mean()
            std_score = scores.std()

            model_result[metric_name] = {
                "mean": mean_score,
                "std": std_score,
                "scores": scores,
            }

            print(
                f"{metric_name}: "
                f"{mean_score:.4f} "
                f"+/- {std_score:.4f}"
            )

        results[model_name] = model_result

    return results


# ============================================================
# Holdout evaluation
# ============================================================

def run_holdout_evaluation(
    models,
    X,
    y,
    labels,
):
    """Evaluate models on the same stratified holdout split."""

    print("\n" + "=" * 70)
    print("STRATIFIED HOLDOUT EVALUATION")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    results = {}

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=labels,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"Weighted F1: "
            f"{weighted_f1:.4f}"
        )

        print("\nClassification Report")
        print("-" * 70)
        print(report)

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        display.plot(
            ax=ax,
            xticks_rotation=45,
        )

        ax.set_title(
            f"{model_name} - Confusion Matrix"
        )

        fig.tight_layout()

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        plot_path = (
            PLOTS_DIR
            / f"{safe_name}_confusion_matrix.png"
        )

        fig.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        results[model_name] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": report,
            "confusion_matrix": matrix,
            "plot_path": plot_path,
        }

    return results


# ============================================================
# Save report
# ============================================================

def save_results(
    df,
    feature_columns,
    label_encoder,
    cv_results,
    holdout_results,
):
    """Save complete Phase 13 evaluation report."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "BioSense AI - Phase 13 ML Evaluation\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write("Dataset Summary\n")
        file.write("-" * 70 + "\n")

        file.write(
            f"Dataset shape: {df.shape}\n"
        )

        file.write(
            f"ML feature count: "
            f"{len(feature_columns)}\n"
        )

        file.write(
            f"Number of classes: "
            f"{len(label_encoder.classes_)}\n"
        )

        file.write(
            f"Cross-validation folds: "
            f"{CV_FOLDS}\n"
        )

        file.write(
            f"Random state: "
            f"{RANDOM_STATE}\n\n"
        )

        file.write("Class Mapping\n")
        file.write("-" * 70 + "\n")

        for index, label in enumerate(
            label_encoder.classes_
        ):

            file.write(
                f"{index} -> {label}\n"
            )

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        file.write("\n")
        file.write("=" * 70 + "\n")
        file.write(
            "5-FOLD STRATIFIED CROSS-VALIDATION\n"
        )
        file.write("=" * 70 + "\n\n")

        for model_name, metrics in cv_results.items():

            file.write(
                f"{model_name}\n"
            )

            file.write(
                "-" * 70 + "\n"
            )

            for metric_name, values in metrics.items():

                file.write(
                    f"{metric_name}: "
                    f"{values['mean']:.4f} "
                    f"+/- "
                    f"{values['std']:.4f}\n"
                )

            file.write("\n")

        # ----------------------------------------------------
        # Holdout
        # ----------------------------------------------------

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "STRATIFIED HOLDOUT EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for model_name, result in (
            holdout_results.items()
        ):

            file.write(
                f"{model_name}\n"
            )

            file.write(
                "-" * 70 + "\n"
            )

            file.write(
                f"Accuracy: "
                f"{result['accuracy']:.4f}\n"
            )

            file.write(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy']:.4f}\n"
            )

            file.write(
                f"Macro F1: "
                f"{result['macro_f1']:.4f}\n"
            )

            file.write(
                f"Weighted F1: "
                f"{result['weighted_f1']:.4f}\n\n"
            )

            file.write(
                "Classification Report\n"
            )

            file.write(
                result["classification_report"]
            )

            file.write("\n\n")


# ============================================================
# Main
# ============================================================

def main():

    df = load_dataset()

    (
        X,
        y,
        label_encoder,
        feature_columns,
    ) = prepare_data(df)

    models = create_models(
        number_of_classes=len(
            label_encoder.classes_
        )
    )

    cv_results = run_cross_validation(
        models,
        X,
        y,
    )

    holdout_results = run_holdout_evaluation(
        models,
        X,
        y,
        label_encoder.classes_,
    )

    save_results(
        df,
        feature_columns,
        label_encoder,
        cv_results,
        holdout_results,
    )

    print("\n" + "=" * 70)
    print("PHASE 13 EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nPlots saved to:\n"
        f"{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()