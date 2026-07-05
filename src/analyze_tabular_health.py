from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
CLEANED = OUTPUTS / "cleaned"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "tabular_health_analysis.log"


DATASETS = {
    "cdc_diabetes": {
        "path": ROOT / "datasets" / "uci_891_cdc_diabetes_health_indicators" / "data.csv",
        "target": "Diabetes_binary",
        "positive_label": 1,
        "selected": ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Income"],
        "categorical": [
            "HighBP",
            "HighChol",
            "CholCheck",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "DiffWalk",
            "Sex",
            "Education",
        ],
    },
    "breast_cancer": {
        "path": ROOT / "datasets" / "uci_17_breast_cancer_wisconsin_diagnostic" / "data.csv",
        "target": "Diagnosis",
        "positive_label": "M",
        "selected": ["radius1", "texture1", "perimeter1", "area1", "smoothness1", "concavity1"],
        "categorical": [],
    },
}


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


def summarize_numeric(df: pd.DataFrame, dataset_name: str, state: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "state": state,
                "variable": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "variance": series.var(ddof=1),
                "std": series.std(ddof=1),
                "cv": series.std(ddof=1) / series.mean() if series.mean() != 0 else np.nan,
                "p25": series.quantile(0.25),
                "p75": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
            }
        )
    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame, selected: list[str], target: str) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if column == target:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            mode = cleaned[column].mode()
            cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "unknown")

    for column in selected:
        series = pd.to_numeric(cleaned[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            cleaned[column] = series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return cleaned


def save_frequency_tables(df: pd.DataFrame, dataset_name: str, target: str, categorical: list[str]) -> None:
    for column in [target] + categorical:
        if column not in df.columns:
            continue
        freq = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="absolute_frequency")
        freq["relative_frequency"] = freq["absolute_frequency"] / len(df)
        freq.to_csv(TABLES / f"{dataset_name}_frequency_{column}.csv", index=False)


def save_plots(df: pd.DataFrame, cleaned: pd.DataFrame, dataset_name: str, config: dict[str, object]) -> None:
    target = str(config["target"])
    selected = list(config["selected"])
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target, hue=target, palette="Set2", legend=False)
    plt.title(f"{dataset_name} target distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{dataset_name}_target_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.ravel(), selected):
        sns.histplot(cleaned[column], bins=25, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(column)
    fig.suptitle(f"{dataset_name} selected variable distributions")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset_name}_selected_histograms.png", dpi=180)
    plt.close(fig)

    first_numeric = selected[0]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=cleaned, x=target, y=first_numeric, hue=target, palette="Set3", legend=False)
    plt.title(f"{dataset_name}: {first_numeric} by target")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{dataset_name}_{first_numeric}_boxplot_by_target.png", dpi=180)
    plt.close()

    corr_cols = [column for column in selected if pd.api.types.is_numeric_dtype(cleaned[column])]
    if len(corr_cols) >= 2:
        plt.figure(figsize=(9, 7))
        sns.heatmap(cleaned[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title(f"{dataset_name} selected correlation heatmap")
        plt.tight_layout()
        plt.savefig(FIGURES / f"{dataset_name}_correlation_heatmap.png", dpi=180)
        plt.close()


def analyze_one(dataset_name: str, config: dict[str, object]) -> dict[str, object]:
    path = Path(config["path"])
    target = str(config["target"])
    selected = list(config["selected"])
    categorical = list(config["categorical"])

    df = pd.read_csv(path)
    cleaned = clean_dataset(df, selected, target)
    cleaned.to_csv(CLEANED / f"{dataset_name}_cleaned.csv", index=False)

    missing = pd.DataFrame(
        {
            "dataset": dataset_name,
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(4).values,
        }
    )
    missing.to_csv(TABLES / f"{dataset_name}_missing_values.csv", index=False)

    raw_summary = summarize_numeric(df, dataset_name, "raw", selected)
    clean_summary = summarize_numeric(cleaned, dataset_name, "cleaned", selected)
    pd.concat([raw_summary, clean_summary], ignore_index=True).to_csv(
        TABLES / f"{dataset_name}_summary.csv", index=False
    )

    variable_types = []
    for column in df.columns:
        if column == target:
            vtype = "target"
        elif column in categorical:
            vtype = "qualitative"
        elif pd.api.types.is_numeric_dtype(df[column]):
            vtype = "quantitative_continuous_or_ordinal"
        else:
            vtype = "qualitative"
        variable_types.append({"dataset": dataset_name, "variable": column, "type": vtype})
    pd.DataFrame(variable_types).to_csv(TABLES / f"{dataset_name}_variable_types.csv", index=False)

    save_frequency_tables(df, dataset_name, target, categorical)
    save_plots(df, cleaned, dataset_name, config)

    logging.info("%s raw shape: %s", dataset_name, df.shape)
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "target": target,
        "cleaned_path": f"outputs/cleaned/{dataset_name}_cleaned.csv",
    }


def main() -> None:
    ensure_dirs()
    setup_logging()
    rows = [analyze_one(name, config) for name, config in DATASETS.items()]
    pd.DataFrame(rows).to_csv(TABLES / "tabular_health_analysis_index.csv", index=False)
    print("Tabular health analysis complete.")


if __name__ == "__main__":
    main()
