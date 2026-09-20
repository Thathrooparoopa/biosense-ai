from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "S-OH"
DATA_DIR = DATASET_ROOT / "data"
METADATA_PATH = DATASET_ROOT / "metadata.csv"

REPORT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "dataset_exploration_report.txt"
)


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv not found:\n{METADATA_PATH}"
        )

    return pd.read_csv(METADATA_PATH)


def inspect_metadata(metadata: pd.DataFrame) -> list[str]:
    lines = []

    lines.append("=" * 70)
    lines.append("METADATA EXPLORATION")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Number of rows: {len(metadata)}")
    lines.append(f"Number of columns: {len(metadata.columns)}")
    lines.append("")

    lines.append("Column names:")
    for column in metadata.columns:
        lines.append(f"  - {column}")

    lines.append("")

    lines.append("Data types:")
    for column, dtype in metadata.dtypes.items():
        lines.append(f"  - {column}: {dtype}")

    lines.append("")

    lines.append("Missing values:")
    missing = metadata.isnull().sum()

    for column, count in missing.items():
        lines.append(f"  - {column}: {count}")

    lines.append("")

    lines.append("First five rows:")
    lines.append(metadata.head().to_string(index=False))

    lines.append("")

    lines.append("Descriptive statistics:")
    lines.append(metadata.describe(include="all").to_string())

    lines.append("")

    return lines


def inspect_diagnosis_distribution(
    metadata: pd.DataFrame,
) -> list[str]:
    lines = []

    lines.append("=" * 70)
    lines.append("DIAGNOSIS DISTRIBUTION")
    lines.append("=" * 70)
    lines.append("")

    possible_columns = [
        "Diagnosis",
        "diagnosis",
        "Diagnosis_code",
        "diagnosis_code",
        "diag",
        "patient_diag_class",
    ]

    diagnosis_column = None

    for column in possible_columns:
        if column in metadata.columns:
            diagnosis_column = column
            break

    if diagnosis_column is None:
        lines.append(
            "No standard diagnosis column was automatically detected."
        )
        lines.append(
            f"Available columns: {list(metadata.columns)}"
        )
        lines.append("")
        return lines

    lines.append(
        f"Diagnosis column detected: {diagnosis_column}"
    )
    lines.append("")

    counts = metadata[diagnosis_column].value_counts(dropna=False)

    for diagnosis, count in counts.items():
        percentage = (count / len(metadata)) * 100

        lines.append(
            f"  {diagnosis}: {count} "
            f"({percentage:.2f}%)"
        )

    lines.append("")

    return lines


def inspect_dataset_directories() -> list[str]:
    lines = []

    lines.append("=" * 70)
    lines.append("DATA DIRECTORY STRUCTURE")
    lines.append("=" * 70)
    lines.append("")

    if not DATA_DIR.exists():
        lines.append(f"Data directory not found: {DATA_DIR}")
        return lines

    diagnosis_directories = sorted(
        [
            directory
            for directory in DATA_DIR.iterdir()
            if directory.is_dir()
            and not directory.name.startswith(".")
        ]
    )

    lines.append(
        f"Number of diagnosis directories: "
        f"{len(diagnosis_directories)}"
    )
    lines.append("")

    total_files = 0

    for directory in diagnosis_directories:
        files = [
            file
            for file in directory.rglob("*")
            if file.is_file()
        ]

        total_files += len(files)

        lines.append(
            f"  {directory.name}: {len(files)} files"
        )

    lines.append("")
    lines.append(f"Total data files: {total_files}")
    lines.append("")

    return lines


def find_first_patient_file() -> Path | None:
    if not DATA_DIR.exists():
        return None

    patient_files = sorted(
        [
            file
            for file in DATA_DIR.rglob("*")
            if file.is_file()
            and file.suffix.lower() == ".json"
        ]
    )

    if not patient_files:
        return None

    return patient_files[0]


def inspect_patient_file() -> list[str]:
    lines = []

    lines.append("=" * 70)
    lines.append("PATIENT FILE EXPLORATION")
    lines.append("=" * 70)
    lines.append("")

    patient_file = find_first_patient_file()

    if patient_file is None:
        lines.append(
            "No JSON patient file was found."
        )
        lines.append("")
        return lines

    lines.append(
        f"Sample patient file:\n{patient_file}"
    )
    lines.append("")

    try:
        with open(
            patient_file,
            "r",
            encoding="utf-8",
        ) as file:
            patient = json.load(file)

    except Exception as exception:
        lines.append(
            f"Could not parse JSON file: {exception}"
        )
        lines.append("")
        return lines

    if isinstance(patient, dict):
        lines.append("Top-level JSON keys:")

        for key in patient.keys():
            lines.append(f"  - {key}")

        lines.append("")

        lines.append("Top-level value types:")

        for key, value in patient.items():
            lines.append(
                f"  - {key}: {type(value).__name__}"
            )

        lines.append("")

        if "sensors" in patient:
            sensors = patient["sensors"]

            lines.append(
                f"Number of sensor groups: {len(sensors)}"
            )
            lines.append("")

            for sensor in sensors:
                if not isinstance(sensor, dict):
                    continue

                sensor_id = sensor.get("id")
                sample_rate = sensor.get("sampleRate")
                channels = sensor.get("channels", [])

                lines.append(
                    f"Sensor group: {sensor_id}"
                )
                lines.append(
                    f"  Sample rate: {sample_rate}"
                )
                lines.append(
                    f"  Channels: {len(channels)}"
                )

                for channel in channels:
                    if not isinstance(channel, dict):
                        continue

                    channel_id = channel.get("id")
                    samples = channel.get("samples", [])

                    lines.append(
                        f"    {channel_id}: "
                        f"{len(samples)} samples"
                    )

                lines.append("")

    else:
        lines.append(
            f"Unexpected JSON root type: "
            f"{type(patient).__name__}"
        )

    return lines


def inspect_enose_channels() -> list[str]:
    lines = []

    lines.append("=" * 70)
    lines.append("eNOSE CHANNEL INSPECTION")
    lines.append("=" * 70)
    lines.append("")

    patient_file = find_first_patient_file()

    if patient_file is None:
        lines.append("No patient JSON file available.")
        lines.append("")
        return lines

    try:
        with open(
            patient_file,
            "r",
            encoding="utf-8",
        ) as file:
            patient = json.load(file)

    except Exception as exception:
        lines.append(
            f"Could not read patient file: {exception}"
        )
        lines.append("")
        return lines

    sensors = patient.get("sensors", [])

    enose_sensors = [
        sensor
        for sensor in sensors
        if isinstance(sensor, dict)
        and str(sensor.get("id", "")).lower()
        in {"enose", "e-nose", "electronic_nose"}
    ]

    if not enose_sensors:
        lines.append(
            "No eNose sensor group was automatically identified."
        )
        lines.append("")

        lines.append("Available sensor groups:")

        for sensor in sensors:
            if isinstance(sensor, dict):
                lines.append(
                    f"  - {sensor.get('id')}"
                )

        lines.append("")
        return lines

    for sensor in enose_sensors:
        channels = sensor.get("channels", [])

        lines.append(
            f"Detected eNose channels: {len(channels)}"
        )
        lines.append("")

        for channel in channels:
            if not isinstance(channel, dict):
                continue

            channel_id = channel.get("id")
            samples = channel.get("samples", [])

            numeric_values = [
                value
                for value in samples
                if isinstance(value, (int, float))
            ]

            lines.append(
                f"Channel: {channel_id}"
            )
            lines.append(
                f"  Total samples: {len(samples)}"
            )
            lines.append(
                f"  Numeric samples: {len(numeric_values)}"
            )

            if numeric_values:
                lines.append(
                    f"  Minimum: {min(numeric_values)}"
                )
                lines.append(
                    f"  Maximum: {max(numeric_values)}"
                )

                series = pd.Series(numeric_values)

                lines.append(
                    f"  Mean: {series.mean()}"
                )
                lines.append(
                    f"  Standard deviation: "
                    f"{series.std()}"
                )

            lines.append("")

    return lines


def save_report(lines: list[str]) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))

    print()
    print("Exploration report saved:")
    print(REPORT_PATH)


def main() -> None:
    print("BioSense AI - S-OH Dataset Exploration")
    print("=" * 70)

    metadata = load_metadata()

    report = []

    report.append(
        "BioSense AI - S-OH Dataset Exploration Report"
    )
    report.append(
        "=" * 70
    )
    report.append("")

    report.extend(
        inspect_metadata(metadata)
    )

    report.extend(
        inspect_diagnosis_distribution(metadata)
    )

    report.extend(
        inspect_dataset_directories()
    )

    report.extend(
        inspect_patient_file()
    )

    report.extend(
        inspect_enose_channels()
    )

    save_report(report)

    print()
    print(
        "Dataset exploration completed successfully."
    )


if __name__ == "__main__":
    main()