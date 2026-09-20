import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "S-OH"
DATA_DIR = DATASET_ROOT / "data"
METADATA_PATH = DATASET_ROOT / "metadata.csv"

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

OUTPUT_CSV = PROCESSED_DIR / "soh_binary_features.csv"
REPORT_PATH = PROCESSED_DIR / "preprocessing_report.txt"


# ============================================================
# Configuration
# ============================================================

TARGET_DIAGNOSES = {
    "Z00": 0,
    "C34": 1,
}

TARGET_NAMES = {
    0: "Healthy Control",
    1: "Lung Cancer",
}

ENOSE_CHANNELS = [f"R{i}" for i in range(1, 18)]

# Used for reporting only.
# We do not reject a patient solely because a channel
# is shorter than this value.
SHORT_SEQUENCE_THRESHOLD = 300


# ============================================================
# Utility functions
# ============================================================

def find_patient_file(
    diagnosis: str,
    patient_id: int,
) -> Path:
    """
    Locate the patient JSON file for a given diagnosis
    and patient ID.
    """

    diagnosis_dir = DATA_DIR / diagnosis

    expected_file = (
        diagnosis_dir / f"patient_{patient_id}.json"
    )

    if expected_file.exists():
        return expected_file

    matches = list(
        diagnosis_dir.glob(
            f"patient_{patient_id}.*"
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Patient file not found: "
        f"diagnosis={diagnosis}, "
        f"patient_id={patient_id}"
    )


def load_patient_json(path: Path) -> dict:
    """
    Load one patient JSON file.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_enose_channels(
    patient_data: dict,
) -> dict[str, np.ndarray]:
    """
    Extract R1-R17 from the eNose sensor group.
    """

    sensors = patient_data.get("sensors", [])

    enose_sensor = None

    for sensor in sensors:
        if (
            isinstance(sensor, dict)
            and sensor.get("id") == "enose"
        ):
            enose_sensor = sensor
            break

    if enose_sensor is None:
        raise ValueError(
            "enose sensor group not found"
        )

    extracted = {}

    for channel in enose_sensor.get(
        "channels",
        [],
    ):
        if not isinstance(channel, dict):
            continue

        channel_id = channel.get("id")

        if channel_id not in ENOSE_CHANNELS:
            continue

        samples = channel.get(
            "samples",
            [],
        )

        values = np.asarray(
            samples,
            dtype=float,
        )

        extracted[channel_id] = values

    missing_channels = [
        channel
        for channel in ENOSE_CHANNELS
        if channel not in extracted
    ]

    if missing_channels:
        raise ValueError(
            f"Missing eNose channels: "
            f"{missing_channels}"
        )

    return extracted


def calculate_features(
    channel_name: str,
    values: np.ndarray,
) -> dict[str, float]:
    """
    Calculate statistical features from one
    sensor time series.

    No artificial padding or interpolation is used.
    """

    if values.size == 0:
        raise ValueError(
            f"{channel_name} contains no samples"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{channel_name} contains NaN or "
            f"infinite values"
        )

    differences = np.diff(values)

    features = {
        f"{channel_name}_sample_count": float(
            len(values)
        ),

        f"{channel_name}_mean": float(
            np.mean(values)
        ),

        f"{channel_name}_std": float(
            np.std(values)
        ),

        f"{channel_name}_min": float(
            np.min(values)
        ),

        f"{channel_name}_max": float(
            np.max(values)
        ),

        f"{channel_name}_median": float(
            np.median(values)
        ),

        f"{channel_name}_range": float(
            np.ptp(values)
        ),

        f"{channel_name}_first": float(
            values[0]
        ),

        f"{channel_name}_last": float(
            values[-1]
        ),

        f"{channel_name}_mean_abs_diff": (
            float(
                np.mean(np.abs(differences))
            )
            if differences.size > 0
            else 0.0
        ),

        f"{channel_name}_diff_std": (
            float(
                np.std(differences)
            )
            if differences.size > 0
            else 0.0
        ),
    }

    return features


def process_patient(
    metadata_row: pd.Series,
) -> tuple[dict, dict]:
    """
    Process one patient.

    Returns:
        features:
            Patient-level ML feature dictionary.

        diagnostics:
            Preprocessing diagnostics for the patient.
    """

    patient_id = int(
        metadata_row["Patient_id"]
    )

    diagnosis = str(
        metadata_row["Diagnosis"]
    )

    patient_path = find_patient_file(
        diagnosis=diagnosis,
        patient_id=patient_id,
    )

    patient_data = load_patient_json(
        patient_path
    )

    channels = extract_enose_channels(
        patient_data
    )

    features = {
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "target": TARGET_DIAGNOSES[diagnosis],
    }

    diagnostics = {
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "file": str(
            patient_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "valid": True,
        "channel_lengths": {},
        "short_channels": [],
    }

    for channel_name in ENOSE_CHANNELS:
        values = channels[channel_name]

        sample_count = len(values)

        diagnostics["channel_lengths"][
            channel_name
        ] = sample_count

        if sample_count < SHORT_SEQUENCE_THRESHOLD:
            diagnostics["short_channels"].append(
                {
                    "channel": channel_name,
                    "sample_count": sample_count,
                }
            )

        channel_features = calculate_features(
            channel_name=channel_name,
            values=values,
        )

        features.update(
            channel_features
        )

    return features, diagnostics


# ============================================================
# Main preprocessing pipeline
# ============================================================

def main():
    print("=" * 70)
    print(
        "BioSense AI - S-O-H Data Preprocessing"
    )
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: "
            f"{DATA_DIR}"
        )

    print(
        f"\nDataset root: {DATASET_ROOT}"
    )
    print(
        f"Metadata: {METADATA_PATH}"
    )
    print(
        f"Data directory: {DATA_DIR}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"\nTotal metadata rows: "
        f"{len(metadata)}"
    )

    required_columns = {
        "Patient_id",
        "Diagnosis",
        "D_class",
        "D_bin_class",
        "Datetime",
        "Week",
        "Site",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required metadata "
            f"columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Filter target classes
    # --------------------------------------------------------

    filtered = metadata[
        metadata["Diagnosis"].isin(
            TARGET_DIAGNOSES.keys()
        )
    ].copy()

    filtered["target"] = (
        filtered["Diagnosis"].map(
            TARGET_DIAGNOSES
        )
    )

    print(
        f"Target rows after filtering: "
        f"{len(filtered)}"
    )

    print("\nTarget distribution:")

    target_counts = (
        filtered["target"]
        .value_counts()
        .sort_index()
    )

    for target, count in (
        target_counts.items()
    ):
        print(
            f"  {target} - "
            f"{TARGET_NAMES[target]}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    processed_rows = []
    diagnostics = []
    failed_patients = []

    short_channel_records = []

    print(
        "\nProcessing patient sensor files..."
    )

    for index, (_, row) in enumerate(
        filtered.iterrows(),
        start=1,
    ):
        patient_id = int(
            row["Patient_id"]
        )

        diagnosis = str(
            row["Diagnosis"]
        )

        try:
            (
                features,
                patient_diagnostics,
            ) = process_patient(row)

            processed_rows.append(
                features
            )

            diagnostics.append(
                patient_diagnostics
            )

            if patient_diagnostics[
                "short_channels"
            ]:
                for short_channel in (
                    patient_diagnostics[
                        "short_channels"
                    ]
                ):
                    short_channel_records.append(
                        {
                            "patient_id": patient_id,
                            "diagnosis": diagnosis,
                            "channel": short_channel[
                                "channel"
                            ],
                            "sample_count": short_channel[
                                "sample_count"
                            ],
                        }
                    )

        except Exception as exc:
            failed_patients.append(
                {
                    "patient_id": patient_id,
                    "diagnosis": diagnosis,
                    "error": str(exc),
                }
            )

        if index % 50 == 0:
            print(
                f"  Processed "
                f"{index}/{len(filtered)} patients"
            )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    processed_df = pd.DataFrame(
        processed_rows
    )

    if processed_df.empty:
        raise RuntimeError(
            "No patients were successfully "
            "processed."
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    identifier_columns = {
        "patient_id",
        "diagnosis",
        "target",
    }

    feature_columns = [
        column
        for column in processed_df.columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_feature_values = int(
        processed_df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_feature_values = int(
        np.isinf(
            processed_df[
                feature_columns
            ].to_numpy()
        ).sum()
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    report_lines = [
        "BioSense AI - S-O-H Preprocessing Report",
        "=" * 70,
        "",
        f"Source dataset: {DATASET_ROOT}",
        f"Metadata rows: {len(metadata)}",
        f"Target rows: {len(filtered)}",
        (
            "Successfully processed: "
            f"{len(processed_df)}"
        ),
        (
            "Failed patients: "
            f"{len(failed_patients)}"
        ),
        "",
        "Target mapping:",
        "  0 = Z00 = Healthy Control",
        "  1 = C34 = Lung Cancer",
        "",
        "Primary sensor channels:",
        ", ".join(ENOSE_CHANNELS),
        "",
        (
            "Number of feature columns: "
            f"{len(feature_columns)}"
        ),
        "",
        "Feature extraction per channel:",
        "  - sample count",
        "  - mean",
        "  - standard deviation",
        "  - minimum",
        "  - maximum",
        "  - median",
        "  - range",
        "  - first value",
        "  - last value",
        "  - mean absolute first difference",
        "  - standard deviation of first differences",
        "",
        "Target distribution:",
    ]

    processed_target_counts = (
        processed_df["target"]
        .value_counts()
        .sort_index()
    )

    for target, count in (
        processed_target_counts.items()
    ):
        report_lines.append(
            f"  {target} - "
            f"{TARGET_NAMES[target]}: "
            f"{count}"
        )

    report_lines.extend(
        [
            "",
            "Feature validation:",
            (
                "  Missing feature values: "
                f"{missing_feature_values}"
            ),
            (
                "  Infinite feature values: "
                f"{infinite_feature_values}"
            ),
            "",
            "Short channel observations:",
            (
                "  Threshold: "
                f"< {SHORT_SEQUENCE_THRESHOLD} samples"
            ),
            (
                "  Total short-channel observations: "
                f"{len(short_channel_records)}"
            ),
        ]
    )

    if short_channel_records:
        for record in short_channel_records:
            report_lines.append(
                f"  Patient "
                f"{record['patient_id']} "
                f"({record['diagnosis']}), "
                f"{record['channel']}: "
                f"{record['sample_count']} samples"
            )
    else:
        report_lines.append(
            "  None"
        )

    report_lines.extend(
        [
            "",
            "Output:",
            f"  {OUTPUT_CSV}",
            "",
            "Failed patients:",
        ]
    )

    if failed_patients:
        for failed in failed_patients:
            report_lines.append(
                f"  Patient "
                f"{failed['patient_id']} "
                f"({failed['diagnosis']}): "
                f"{failed['error']}"
            )
    else:
        report_lines.append(
            "  None"
        )

    report_lines.extend(
        [
            "",
            "Processing policy:",
            "  - One row represents one patient.",
            "  - Only R1-R17 are used as the primary sensor inputs.",
            "  - No artificial sensor samples are created.",
            "  - Short but valid sensor sequences are retained.",
            "  - Sample counts are recorded as features.",
            "  - Patient_id is retained only as an identifier.",
            "  - Diagnosis and target are retained for labels.",
            "  - Age, gender, site, week and datetime are not used as model features.",
            "  - Auxiliary sensors are excluded from this first baseline dataset.",
            "  - No machine-learning model is trained in this phase.",
            "",
            "Medical disclaimer:",
            "This application is an experimental research prototype and is not",
            "a medical diagnostic device. Its results must not be used to diagnose,",
            "treat, or rule out cancer. Medical decisions should be made only by",
            "qualified healthcare professionals.",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "PREPROCESSING COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        f"\nSuccessfully processed: "
        f"{len(processed_df)}"
    )

    print(
        f"Failed patients: "
        f"{len(failed_patients)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Short channel observations: "
        f"{len(short_channel_records)}"
    )

    print(
        "\nTarget distribution:"
    )

    for target, count in (
        processed_df["target"]
        .value_counts()
        .sort_index()
        .items()
    ):
        print(
            f"  {target} - "
            f"{TARGET_NAMES[target]}: "
            f"{count}"
        )

    print(
        "\nMissing feature values: "
        f"{missing_feature_values}"
    )

    print(
        "Infinite feature values: "
        f"{infinite_feature_values}"
    )

    print(
        "\nSaved dataset:"
    )
    print(OUTPUT_CSV)

    print(
        "\nSaved report:"
    )
    print(REPORT_PATH)


if __name__ == "__main__":
    main()