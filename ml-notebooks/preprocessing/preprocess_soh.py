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

OUTPUT_CSV = (
    PROCESSED_DIR / "soh_multiclass_features.csv"
)

REPORT_PATH = (
    PROCESSED_DIR / "preprocessing_report.txt"
)


# ============================================================
# Configuration
# ============================================================

ENOSE_CHANNELS = [
    f"R{i}"
    for i in range(1, 18)
]

SHORT_SEQUENCE_THRESHOLD = 300


# ============================================================
# Utility functions
# ============================================================

def find_patient_file(
    diagnosis: str,
    patient_id: int,
) -> Path:

    diagnosis_dir = DATA_DIR / diagnosis

    expected_file = (
        diagnosis_dir
        / f"patient_{patient_id}.json"
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


def load_patient_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_enose_channels(
    patient_data: dict,
) -> dict[str, np.ndarray]:

    sensors = patient_data.get(
        "sensors",
        []
    )

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
            "eNose sensor group not found"
        )

    extracted = {}

    for channel in enose_sensor.get(
        "channels",
        [],
    ):

        if not isinstance(
            channel,
            dict,
        ):
            continue

        channel_id = channel.get(
            "id"
        )

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
            "Missing eNose channels: "
            f"{missing_channels}"
        )

    return extracted


def calculate_features(
    channel_name: str,
    values: np.ndarray,
) -> dict[str, float]:

    if values.size == 0:

        raise ValueError(
            f"{channel_name} contains no samples"
        )

    if not np.isfinite(values).all():

        raise ValueError(
            f"{channel_name} contains "
            "NaN or infinite values"
        )

    differences = np.diff(values)

    return {

        f"{channel_name}_sample_count":
            float(len(values)),

        f"{channel_name}_mean":
            float(np.mean(values)),

        f"{channel_name}_std":
            float(np.std(values)),

        f"{channel_name}_min":
            float(np.min(values)),

        f"{channel_name}_max":
            float(np.max(values)),

        f"{channel_name}_median":
            float(np.median(values)),

        f"{channel_name}_range":
            float(np.ptp(values)),

        f"{channel_name}_first":
            float(values[0]),

        f"{channel_name}_last":
            float(values[-1]),

        f"{channel_name}_mean_abs_diff":
            (
                float(
                    np.mean(
                        np.abs(differences)
                    )
                )
                if differences.size > 0
                else 0.0
            ),

        f"{channel_name}_diff_std":
            (
                float(
                    np.std(differences)
                )
                if differences.size > 0
                else 0.0
            ),
    }


def process_patient(
    metadata_row: pd.Series,
) -> tuple[dict, dict]:

    patient_id = int(
        metadata_row["Patient_id"]
    )

    diagnosis = str(
        metadata_row["Diagnosis"]
    )

    patient_path = find_patient_file(
        diagnosis,
        patient_id,
    )

    patient_data = load_patient_json(
        patient_path
    )

    channels = extract_enose_channels(
        patient_data
    )

    features = {

        "patient_id":
            patient_id,

        "diagnosis":
            diagnosis,
    }

    diagnostics = {

        "patient_id":
            patient_id,

        "diagnosis":
            diagnosis,

        "file":
            str(
                patient_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "short_channels":
            [],
    }

    for channel_name in ENOSE_CHANNELS:

        values = channels[channel_name]

        sample_count = len(values)

        if (
            sample_count
            < SHORT_SEQUENCE_THRESHOLD
        ):

            diagnostics[
                "short_channels"
            ].append(
                {
                    "channel":
                        channel_name,

                    "sample_count":
                        sample_count,
                }
            )

        channel_features = (
            calculate_features(
                channel_name,
                values,
            )
        )

        features.update(
            channel_features
        )

    return (
        features,
        diagnostics,
    )


# ============================================================
# Main preprocessing
# ============================================================

def main():

    print("=" * 70)
    print(
        "BioSense AI - Multi-Class S-O-H Preprocessing"
    )
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_PATH}"
        )

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

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print(
        "\nDiagnosis distribution:"
    )

    diagnosis_counts = (
        metadata["Diagnosis"]
        .value_counts()
        .sort_index()
    )

    for diagnosis, count in (
        diagnosis_counts.items()
    ):

        print(
            f"  {diagnosis}: {count}"
        )

    # --------------------------------------------------------
    # Process all patients
    # --------------------------------------------------------

    processed_rows = []

    failed_patients = []

    short_channel_records = []

    print(
        "\nProcessing patient files..."
    )

    for index, (_, row) in enumerate(
        metadata.iterrows(),
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
                diagnostics,
            ) = process_patient(row)

            processed_rows.append(
                features
            )

            for short_channel in (
                diagnostics[
                    "short_channels"
                ]
            ):

                short_channel_records.append(
                    {
                        "patient_id":
                            patient_id,

                        "diagnosis":
                            diagnosis,

                        "channel":
                            short_channel[
                                "channel"
                            ],

                        "sample_count":
                            short_channel[
                                "sample_count"
                            ],
                    }
                )

        except Exception as exc:

            failed_patients.append(
                {
                    "patient_id":
                        patient_id,

                    "diagnosis":
                        diagnosis,

                    "error":
                        str(exc),
                }
            )

        if index % 100 == 0:

            print(
                f"  Processed "
                f"{index}/{len(metadata)}"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    processed_df = pd.DataFrame(
        processed_rows
    )

    if processed_df.empty:

        raise RuntimeError(
            "No patients were processed."
        )

    feature_columns = [
        column
        for column in processed_df.columns
        if column not in {
            "patient_id",
            "diagnosis",
        }
    ]

    missing_values = int(
        processed_df[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    infinite_values = int(
        np.isinf(
            processed_df[
                feature_columns
            ].to_numpy()
        ).sum()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    processed_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = [

        "BioSense AI - S-O-H Multi-Class "
        "Preprocessing Report",

        "=" * 70,

        "",

        f"Metadata rows: {len(metadata)}",

        (
            "Successfully processed: "
            f"{len(processed_df)}"
        ),

        (
            "Failed patients: "
            f"{len(failed_patients)}"
        ),

        "",

        "Research task:",

        (
            "Multi-class classification across "
            "all diagnostic groups in S-O-H."
        ),

        "",

        "Primary sensor channels:",

        ", ".join(ENOSE_CHANNELS),

        "",

        (
            "Number of feature columns: "
            f"{len(feature_columns)}"
        ),

        "",

        "Diagnosis distribution:",
    ]

    processed_counts = (
        processed_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    for diagnosis, count in (
        processed_counts.items()
    ):

        report.append(
            f"  {diagnosis}: {count}"
        )

    report.extend(
        [

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

            "Feature validation:",

            (
                "  Missing feature values: "
                f"{missing_values}"
            ),

            (
                "  Infinite feature values: "
                f"{infinite_values}"
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

            report.append(
                f"  Patient "
                f"{record['patient_id']} "
                f"({record['diagnosis']}), "
                f"{record['channel']}: "
                f"{record['sample_count']} samples"
            )

    else:

        report.append(
            "  None"
        )

    report.extend(
        [

            "",

            "Failed patients:",
        ]
    )

    if failed_patients:

        for failed in failed_patients:

            report.append(
                f"  Patient "
                f"{failed['patient_id']} "
                f"({failed['diagnosis']}): "
                f"{failed['error']}"
            )

    else:

        report.append(
            "  None"
        )

    report.extend(
        [

            "",

            "Processing policy:",

            "  - One row represents one patient.",

            "  - All diagnostic groups are retained.",

            "  - Only R1-R17 are used as primary sensor inputs.",

            "  - No artificial sensor samples are created.",

            "  - Short valid recordings are retained.",

            "  - Sample counts are recorded.",

            "  - Patient_id is an identifier only.",

            "  - Diagnosis is the classification label.",

            "  - Age, gender, site, week and datetime are not model features.",

            "  - Auxiliary sensors are excluded from the first baseline.",

            "  - No machine-learning model is trained in this phase.",

            "",

            "Output:",

            f"  {OUTPUT_CSV}",

            "",

            "Medical disclaimer:",

            "This application is an experimental research prototype "
            "and is not a medical diagnostic device.",

            "Its results must not be used to diagnose, treat, "
            "or rule out cancer.",

            "Medical decisions should be made only by qualified "
            "healthcare professionals.",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-CLASS PREPROCESSING COMPLETE"
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
        "\nDiagnosis distribution:"
    )

    print(
        processed_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nMissing feature values: "
        f"{missing_values}"
    )

    print(
        "Infinite feature values: "
        f"{infinite_values}"
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