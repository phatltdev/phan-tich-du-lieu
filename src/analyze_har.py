from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "har_analysis.log"


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, filemode="w", format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["activity_id", "activity"])
    parts = []
    for split in ["train", "test"]:
        y = pd.read_csv(DATA_DIR / split / f"y_{split}.txt", header=None, names=["activity_id"])
        subject = pd.read_csv(DATA_DIR / split / f"subject_{split}.txt", header=None, names=["subject"])
        part = pd.concat([subject, y], axis=1)
        part["split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).merge(labels, on="activity_id", how="left")
    return df


def load_feature_sample(nrows: int = 3000) -> pd.DataFrame:
    features = pd.read_csv(DATA_DIR / "features.txt", sep=r"\s+", header=None, names=["feature_id", "feature"])
    names = [f"f{i:03d}_{name}" for i, name in enumerate(features["feature"].tolist(), start=1)]
    train = pd.read_csv(DATA_DIR / "train" / "X_train.txt", sep=r"\s+", header=None, names=names, nrows=nrows)
    y = pd.read_csv(DATA_DIR / "train" / "y_train.txt", header=None, names=["activity_id"], nrows=nrows)
    labels = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["activity_id", "activity"])
    return pd.concat([train, y], axis=1).merge(labels, on="activity_id", how="left")


def main() -> None:
    ensure_dirs()
    setup_logging()
    sns.set_theme(style="whitegrid")

    labels = load_labels()
    freq = labels["activity"].value_counts().rename_axis("activity").reset_index(name="count")
    freq["percent"] = freq["count"] / freq["count"].sum() * 100
    freq.to_csv(TABLES / "har_activity_distribution.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=freq, x="count", y="activity", color="#4C72B0")
    plt.title("HAR activity distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "har_activity_distribution.png", dpi=200)
    plt.close()

    sample = load_feature_sample()
    selected = ["f001_tBodyAcc-mean()-X", "f004_tBodyAcc-std()-X", "f121_tBodyGyro-mean()-X"]
    summary = sample[selected].describe().T.reset_index().rename(columns={"index": "feature"})
    summary.to_csv(TABLES / "har_selected_feature_summary.csv", index=False)

    long = sample[["activity"] + selected].melt(id_vars="activity", var_name="feature", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=long, x="feature", y="value", hue="activity", showfliers=False)
    plt.title("HAR selected features by activity")
    plt.xticks(rotation=15)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "har_selected_feature_boxplot.png", dpi=200)
    plt.close()

    logging.info("HAR rows=%s", len(labels))


if __name__ == "__main__":
    main()
