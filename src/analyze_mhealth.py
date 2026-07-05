from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "MHEALTHDATASET" / "MHEALTHDATASET"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "mhealth_analysis.log"

SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(1, 24)]
COLUMNS = SENSOR_COLUMNS + ["activity"]
ACTIVITY_NAMES = {
    0: "NULL", 1: "Standing still", 2: "Sitting", 3: "Lying", 4: "Walking", 5: "Climbing stairs",
    6: "Waist bends", 7: "Frontal arms", 8: "Knees bending", 9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front/back",
}


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, filemode="w", format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_sample(max_rows_per_file: int = 40000) -> pd.DataFrame:
    parts = []
    for path in sorted(DATA_DIR.glob("mHealth_subject*.log")):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, nrows=max_rows_per_file)
        df["subject"] = path.stem.replace("mHealth_subject", "")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ensure_dirs()
    setup_logging()
    sns.set_theme(style="whitegrid")

    df = load_sample()
    df["activity_name"] = df["activity"].map(ACTIVITY_NAMES).fillna(df["activity"].astype(str))

    freq = df["activity_name"].value_counts().rename_axis("activity").reset_index(name="count")
    freq["percent"] = freq["count"] / freq["count"].sum() * 100
    freq.to_csv(TABLES / "mhealth_activity_distribution_sample.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=freq, x="count", y="activity", color="#55A868")
    plt.title("MHEALTH activity distribution sample")
    plt.tight_layout()
    plt.savefig(FIGURES / "mhealth_activity_distribution.png", dpi=200)
    plt.close()

    selected = ["sensor_01", "sensor_02", "sensor_03", "sensor_10", "sensor_11", "sensor_12"]
    summary = df[selected].describe().T.reset_index().rename(columns={"index": "sensor"})
    summary.to_csv(TABLES / "mhealth_selected_sensor_summary.csv", index=False)

    long = df[df["activity"] != 0][["activity_name"] + selected[:3]].melt(id_vars="activity_name", var_name="sensor", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=long, x="sensor", y="value", hue="activity_name", showfliers=False)
    plt.title("MHEALTH chest acceleration channels by activity")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "mhealth_sensor_boxplot.png", dpi=200)
    plt.close()

    logging.info("MHEALTH sampled rows=%s", len(df))


if __name__ == "__main__":
    main()
