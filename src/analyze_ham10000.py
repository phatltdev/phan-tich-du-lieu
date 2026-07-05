from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "ham10000"
SOURCE_PATH = Path((DATA_DIR / "source_path.txt").read_text(encoding="utf-8").strip())
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
CLEANED = OUTPUTS / "cleaned"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "ham10000_analysis.log"


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, CLEANED, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def find_image_path(image_id: str) -> Path | None:
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        candidate = SOURCE_PATH / part / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


def inspect_images(metadata: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    rows = []
    sample = metadata.sample(min(sample_size, len(metadata)), random_state=42)
    for _, row in sample.iterrows():
        image_path = find_image_path(row["image_id"])
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            arr = np.asarray(image.convert("L"))
            rows.append(
                {
                    "image_id": row["image_id"],
                    "dx": row["dx"],
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                    "mean_intensity": float(arr.mean()),
                    "std_intensity": float(arr.std()),
                }
            )
    return pd.DataFrame(rows)


def save_tables(metadata: pd.DataFrame, image_features: pd.DataFrame) -> None:
    metadata.to_csv(CLEANED / "ham10000_metadata_cleaned.csv", index=False)
    image_features.to_csv(TABLES / "ham10000_image_feature_sample.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "samples": len(metadata),
                "lesions": metadata["lesion_id"].nunique(),
                "classes": metadata["dx"].nunique(),
                "metadata_columns": len(metadata.columns),
                "image_feature_sample": len(image_features),
                "image_source_path": str(SOURCE_PATH),
            }
        ]
    )
    summary.to_csv(TABLES / "ham10000_dataset_summary.csv", index=False)

    for column in ["dx", "dx_type", "sex", "localization"]:
        freq = metadata[column].value_counts(dropna=False).rename_axis(column).reset_index(name="absolute_frequency")
        freq["relative_frequency"] = freq["absolute_frequency"] / len(metadata)
        freq.to_csv(TABLES / f"ham10000_frequency_{column}.csv", index=False)

    age_summary = metadata["age"].describe(percentiles=[0.25, 0.5, 0.75]).to_frame("age").T
    age_summary["iqr"] = age_summary["75%"] - age_summary["25%"]
    age_summary.to_csv(TABLES / "ham10000_age_summary.csv", index=False)


def save_plots(metadata: pd.DataFrame, image_features: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    order = metadata["dx"].value_counts().index
    sns.countplot(data=metadata, x="dx", order=order, hue="dx", palette="Set2", legend=False)
    plt.title("HAM10000 lesion class distribution")
    plt.xlabel("Diagnosis class")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(FIGURES / "ham10000_class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(metadata["age"].dropna(), bins=25, kde=True, color="#4C78A8")
    plt.title("HAM10000 patient age distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(FIGURES / "ham10000_age_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    loc_order = metadata["localization"].value_counts().index
    sns.countplot(data=metadata, y="localization", order=loc_order, hue="localization", palette="Set3", legend=False)
    plt.title("HAM10000 lesion localization distribution")
    plt.xlabel("Image count")
    plt.ylabel("Localization")
    plt.tight_layout()
    plt.savefig(FIGURES / "ham10000_localization_bar.png", dpi=180)
    plt.close()

    if not image_features.empty:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=image_features, x="dx", y="mean_intensity", hue="dx", palette="Set2", legend=False)
        plt.title("HAM10000 sampled mean pixel intensity by class")
        plt.xlabel("Diagnosis class")
        plt.ylabel("Mean grayscale intensity")
        plt.tight_layout()
        plt.savefig(FIGURES / "ham10000_mean_intensity_boxplot.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.histplot(image_features["width"], bins=10, color="#F58518", label="width")
        sns.histplot(image_features["height"], bins=10, color="#54A24B", label="height")
        plt.title("HAM10000 sampled image dimensions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / "ham10000_image_dimensions_histogram.png", dpi=180)
        plt.close()


def main() -> None:
    ensure_dirs()
    setup_logging()
    metadata = pd.read_csv(METADATA_PATH)
    image_features = inspect_images(metadata)
    save_tables(metadata, image_features)
    save_plots(metadata, image_features)
    logging.info("Metadata rows: %s", len(metadata))
    logging.info("Image feature sample rows: %s", len(image_features))
    logging.info("Image source path: %s", SOURCE_PATH)
    print("HAM10000 analysis complete.")


if __name__ == "__main__":
    main()
