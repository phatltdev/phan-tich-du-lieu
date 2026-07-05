from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "uci_891_cdc_diabetes_health_indicators" / "data.csv"
OUTPUTS = ROOT / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
TABLES_DIR = OUTPUTS / "tables"
CLEANED_DIR = OUTPUTS / "cleaned"
LOGS_DIR = OUTPUTS / "logs"
LOG_PATH = LOGS_DIR / "cdc_diabetes_analysis.log"

TARGET = "Diabetes_binary"

VARIABLE_TYPES = {
    "HighBP": ("qualitative_binary", "High blood pressure: 1 yes, 0 no"),
    "HighChol": ("qualitative_binary", "High cholesterol: 1 yes, 0 no"),
    "CholCheck": ("qualitative_binary", "Cholesterol check in past 5 years"),
    "BMI": ("quantitative_continuous", "Body Mass Index"),
    "Smoker": ("qualitative_binary", "Smoked at least 100 cigarettes"),
    "Stroke": ("qualitative_binary", "History of stroke"),
    "HeartDiseaseorAttack": ("qualitative_binary", "Coronary heart disease or myocardial infarction"),
    "PhysActivity": ("qualitative_binary", "Physical activity in past 30 days"),
    "Fruits": ("qualitative_binary", "Consumes fruit at least once per day"),
    "Veggies": ("qualitative_binary", "Consumes vegetables at least once per day"),
    "HvyAlcoholConsump": ("qualitative_binary", "Heavy alcohol consumption"),
    "AnyHealthcare": ("qualitative_binary", "Has healthcare coverage"),
    "NoDocbcCost": ("qualitative_binary", "Could not see doctor because of cost"),
    "GenHlth": ("ordinal", "General health rating: 1 excellent to 5 poor"),
    "MentHlth": ("quantitative_discrete", "Days of poor mental health in past 30 days"),
    "PhysHlth": ("quantitative_discrete", "Days of poor physical health in past 30 days"),
    "DiffWalk": ("qualitative_binary", "Difficulty walking or climbing stairs"),
    "Sex": ("qualitative_binary", "Sex code"),
    "Age": ("ordinal", "Age group code"),
    "Education": ("ordinal", "Education level code"),
    "Income": ("ordinal", "Income level code"),
    "Diabetes_binary": ("qualitative_binary_target", "0 no diabetes, 1 prediabetes/diabetes"),
}

SELECTED_VARIABLES = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
NUMERIC_FOR_STATS = SELECTED_VARIABLES
CLIP_COLUMNS = ["BMI", "MentHlth", "PhysHlth"]
CATEGORICAL_FOR_FREQ = [
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
    "GenHlth",
    "Age",
    "Education",
    "Income",
    TARGET,
]


def ensure_dirs() -> None:
    for directory in [FIGURES_DIR, TABLES_DIR, CLEANED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def describe_numeric(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_FOR_STATS:
        series = df[column].dropna()
        rows.append(
            {
                "dataset_state": prefix,
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
                "p50": series.quantile(0.50),
                "p75": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
            }
        )
    return pd.DataFrame(rows)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in CLIP_COLUMNS:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
    return cleaned


def save_metadata_tables(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    variable_rows = [
        {
            "variable": column,
            "type": VARIABLE_TYPES[column][0],
            "meaning": VARIABLE_TYPES[column][1],
            "selected_for_chapter3": column in SELECTED_VARIABLES,
        }
        for column in df.columns
    ]
    pd.DataFrame(variable_rows).to_csv(TABLES_DIR / "cdc_diabetes_variable_types.csv", index=False)

    missing = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(4).values,
        }
    )
    missing.to_csv(TABLES_DIR / "cdc_diabetes_missing_values.csv", index=False)

    raw_summary = describe_numeric(df, "raw")
    cleaned_summary = describe_numeric(cleaned, "cleaned")
    pd.concat([raw_summary, cleaned_summary], ignore_index=True).to_csv(
        TABLES_DIR / "cdc_diabetes_summary.csv", index=False
    )

    compare = raw_summary.merge(cleaned_summary, on="variable", suffixes=("_raw", "_cleaned"))
    compare[
        [
            "variable",
            "mean_raw",
            "mean_cleaned",
            "std_raw",
            "std_cleaned",
            "iqr_raw",
            "iqr_cleaned",
            "min_raw",
            "min_cleaned",
            "max_raw",
            "max_cleaned",
        ]
    ].to_csv(TABLES_DIR / "cdc_diabetes_preprocessing_comparison.csv", index=False)

    for column in CATEGORICAL_FOR_FREQ:
        freq = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="absolute_frequency")
        freq["relative_frequency"] = freq["absolute_frequency"] / len(df)
        freq.to_csv(TABLES_DIR / f"cdc_diabetes_frequency_{column}.csv", index=False)

    cleaned.to_csv(CLEANED_DIR / "cdc_diabetes_cleaned.csv", index=False)


def save_categorical_by_target(cleaned: pd.DataFrame, column: str) -> None:
    rate = cleaned.groupby(column)[TARGET].mean().reset_index(name="diabetes_rate")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=rate, x=column, y="diabetes_rate", color="#4C78A8")
    plt.title(f"Diabetes rate by {column}")
    plt.ylabel("Mean Diabetes_binary")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cdc_diabetes_{column.lower()}_by_target.png", dpi=180)
    plt.close()


def save_plots(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False)
    plt.title("CDC Diabetes Binary Target Distribution")
    plt.xlabel("0 = no diabetes, 1 = prediabetes/diabetes")
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cdc_diabetes_target_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["BMI"], bins=30, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Raw BMI distribution")
    sns.histplot(cleaned["BMI"], bins=30, kde=True, ax=axes[1], color="#59A14F")
    axes[1].set_title("Cleaned BMI distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cdc_diabetes_bmi_raw_vs_cleaned.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, column in zip(axes.ravel(), SELECTED_VARIABLES):
        sns.histplot(cleaned[column], bins=25, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(column)
    axes.ravel()[-1].axis("off")
    fig.suptitle("CDC Diabetes selected variable distributions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cdc_diabetes_selected_histograms.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=cleaned, x=TARGET, y="BMI", hue=TARGET, palette="Set3", legend=False)
    plt.title("Cleaned BMI by diabetes status")
    plt.xlabel("0 = no diabetes, 1 = prediabetes/diabetes")
    plt.ylabel("BMI")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cdc_diabetes_bmi_boxplot_by_target.png", dpi=180)
    plt.close()

    sample = cleaned.sample(min(len(cleaned), 12000), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="Age", y="BMI", hue=TARGET, palette="Set1", alpha=0.35, s=18)
    plt.title("Age group and BMI by diabetes status")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cdc_diabetes_age_bmi_scatter.png", dpi=180)
    plt.close()

    corr_cols = SELECTED_VARIABLES + [TARGET]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cleaned[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("CDC Diabetes selected correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cdc_diabetes_correlation_heatmap.png", dpi=180)
    plt.close()

    for column in ["HighBP", "HighChol", "PhysActivity"]:
        save_categorical_by_target(cleaned, column)


def save_evidence_index() -> None:
    rows = [
        {"artifact_type": "script", "path": "src/analyze_cdc_diabetes.py", "description": "CDC Diabetes descriptive analysis source"},
        {"artifact_type": "raw_data", "path": "datasets/uci_891_cdc_diabetes_health_indicators/data.csv", "description": "Raw CDC Diabetes Health Indicators data"},
        {"artifact_type": "cleaned_data", "path": "outputs/cleaned/cdc_diabetes_cleaned.csv", "description": "Cleaned CDC Diabetes data"},
        {"artifact_type": "table", "path": "outputs/tables/cdc_diabetes_summary.csv", "description": "Raw and cleaned descriptive statistics"},
        {"artifact_type": "table", "path": "outputs/tables/cdc_diabetes_preprocessing_comparison.csv", "description": "Raw vs cleaned comparison"},
        {"artifact_type": "figure", "path": "outputs/figures/cdc_diabetes_bmi_raw_vs_cleaned.png", "description": "BMI raw vs cleaned histogram"},
        {"artifact_type": "figure", "path": "outputs/figures/cdc_diabetes_correlation_heatmap.png", "description": "Selected variable correlation heatmap"},
        {"artifact_type": "log", "path": "outputs/logs/cdc_diabetes_analysis.log", "description": "Execution log"},
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cdc_diabetes_evidence_index.csv", index=False)


def main() -> None:
    ensure_dirs()
    setup_logging()
    df = load_data()
    cleaned = clean_data(df)
    save_metadata_tables(df, cleaned)
    save_plots(df, cleaned)
    save_evidence_index()
    logging.info("CDC Diabetes raw shape: %s", df.shape)
    logging.info("CDC Diabetes cleaned shape: %s", cleaned.shape)
    print("CDC Diabetes analysis complete.")


if __name__ == "__main__":
    main()
