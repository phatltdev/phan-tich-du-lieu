from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "heart+disease" / "processed.cleveland.data"
OUTPUTS = ROOT / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
TABLES_DIR = OUTPUTS / "tables"
CLEANED_DIR = OUTPUTS / "cleaned"
LOGS_DIR = OUTPUTS / "logs"
LOG_PATH = LOGS_DIR / "heart_disease_analysis.log"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

VARIABLE_TYPES = {
    "age": ("quantitative_discrete", "Age in years"),
    "sex": ("qualitative", "Sex: 1 male, 0 female"),
    "cp": ("qualitative", "Chest pain type"),
    "trestbps": ("quantitative_continuous", "Resting blood pressure"),
    "chol": ("quantitative_continuous", "Serum cholesterol"),
    "fbs": ("qualitative", "Fasting blood sugar > 120 mg/dl"),
    "restecg": ("qualitative", "Resting electrocardiographic result"),
    "thalach": ("quantitative_continuous", "Maximum heart rate achieved"),
    "exang": ("qualitative", "Exercise induced angina"),
    "oldpeak": ("quantitative_continuous", "ST depression induced by exercise"),
    "slope": ("qualitative", "Slope of peak exercise ST segment"),
    "ca": ("quantitative_discrete", "Number of major vessels colored by fluoroscopy"),
    "thal": ("qualitative", "Thalassemia category"),
    "num": ("qualitative", "Heart disease diagnosis: 0 absent, 1-4 present/severity"),
}

SELECTED_VARIABLES = ["age", "trestbps", "chol", "thalach", "oldpeak", "num"]
NUMERIC_FOR_STATS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FOR_FREQ = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "num"]


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
    df = pd.read_csv(DATA_PATH, names=COLUMNS, na_values="?")
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["target_binary"] = (df["num"] > 0).astype(int)
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

    for column in ["ca", "thal"]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in NUMERIC_FOR_STATS:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)

    cleaned["target_binary"] = (cleaned["num"] > 0).astype(int)
    return cleaned


def save_metadata_tables(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    variable_rows = [
        {
            "variable": column,
            "type": VARIABLE_TYPES[column][0],
            "meaning": VARIABLE_TYPES[column][1],
            "selected_for_chapter3": column in SELECTED_VARIABLES,
        }
        for column in COLUMNS
    ]
    pd.DataFrame(variable_rows).to_csv(TABLES_DIR / "heart_disease_variable_types.csv", index=False)

    missing = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(4).values,
        }
    )
    missing.to_csv(TABLES_DIR / "heart_disease_missing_values.csv", index=False)

    raw_summary = describe_numeric(df, "raw")
    cleaned_summary = describe_numeric(cleaned, "cleaned")
    pd.concat([raw_summary, cleaned_summary], ignore_index=True).to_csv(
        TABLES_DIR / "heart_disease_summary.csv", index=False
    )

    compare = raw_summary.merge(
        cleaned_summary,
        on="variable",
        suffixes=("_raw", "_cleaned"),
    )
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
    ].to_csv(TABLES_DIR / "heart_disease_preprocessing_comparison.csv", index=False)

    for column in CATEGORICAL_FOR_FREQ:
        freq = (
            df[column]
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="absolute_frequency")
        )
        freq["relative_frequency"] = freq["absolute_frequency"] / len(df)
        freq.to_csv(TABLES_DIR / f"heart_disease_frequency_{column}.csv", index=False)

    cleaned.to_csv(CLEANED_DIR / "heart_disease_cleaned.csv", index=False)


def save_plots(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="target_binary", hue="target_binary", palette="Set2", legend=False)
    plt.title("Heart Disease Binary Target Distribution")
    plt.xlabel("0 = absent, 1 = present")
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heart_disease_target_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["age"], bins=12, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Raw age distribution")
    sns.histplot(cleaned["age"], bins=12, kde=True, ax=axes[1], color="#59A14F")
    axes[1].set_title("Cleaned age distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heart_disease_age_raw_vs_cleaned.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=cleaned, x="target_binary", y="chol", hue="target_binary", palette="Set3", legend=False)
    plt.title("Cleaned cholesterol by disease presence")
    plt.xlabel("0 = absent, 1 = present")
    plt.ylabel("Serum cholesterol")
    # Aligned y-axis shared with the Excel preview (chol ranges 126-564).
    plt.ylim(0, 600)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heart_disease_cholesterol_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=cleaned, x="age", y="thalach", hue="target_binary", palette="Set1", alpha=0.85)
    plt.title("Age vs maximum heart rate")
    plt.xlabel("Age")
    plt.ylabel("Maximum heart rate achieved")
    # Aligned limits shared with the Excel preview (create_excel_visualizations.py)
    # so the two versions can be compared on the same axes.
    plt.xlim(28, 78)
    plt.ylim(80, 205)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heart_disease_age_thalach_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = cleaned[NUMERIC_FOR_STATS + ["target_binary"]].corr()
    # Aligned color scale and colormap with the Excel preview (RdBu_r, vmin=-1, vmax=1)
    # so both versions read the correlation on the same color axis.
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True)
    plt.title("Heart Disease Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heart_disease_correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="cp", color="#4e79a7")
    plt.title("Heart Disease - Chest pain type frequency")
    plt.xlabel("Chest pain type")
    plt.ylabel("Sample count")
    # Aligned y-axis shared with the Excel preview (max total count is 144 for
    # cp=4). x-axis keeps seaborn's default categorical range so all four cp
    # levels remain visible.
    plt.ylim(0, 150)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heart_disease_chest_pain_bar.png", dpi=180)
    plt.close()


GROUP_STATS_VARIABLES = ["chol", "thalach"]


def describe_by_group(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Compute central, dispersion, and shape statistics per target_binary group.

    Extends the per-group mean/median table with extra measures (std, variance, CV,
    IQR, MAD, range, skewness) so that "statistics by target group" is shown to apply
    beyond mean/median. All values are computed from the raw frame on non-missing
    observations of each variable.
    """
    rows = []
    for variable in variables:
        for group, subset in df.groupby("target_binary"):
            series = subset[variable].dropna()
            mean = series.mean()
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            std = series.std(ddof=1)
            mad = np.median(np.abs(series - median))
            rows.append(
                {
                    "variable": variable,
                    "target_binary": int(group),
                    "group_label": "khong benh" if int(group) == 0 else "co benh",
                    "n": int(series.count()),
                    "mean": round(mean, 4),
                    "median": round(median, 4),
                    "mean_minus_median": round(mean - median, 4),
                    "std": round(std, 4),
                    "variance": round(series.var(ddof=1), 4),
                    "cv": round(std / mean, 4) if mean != 0 else np.nan,
                    "iqr": round(q3 - q1, 4),
                    "mad": round(mad, 4),
                    "range": round(series.max() - series.min(), 4),
                    "min": round(series.min(), 4),
                    "max": round(series.max(), 4),
                    "skew": round(series.skew(), 4),
                    "kurtosis": round(series.kurtosis(), 4),
                }
            )
    return pd.DataFrame(rows)


def cross_tab_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Build a crosstab of a categorical variable vs target_binary with row/col totals
    and within-group proportions, so frequency/mode-style measures by target group are
    also demonstrated (not only mean/median for quantitative variables)."""
    ct = pd.crosstab(df[column], df["target_binary"])
    ct.columns = [f"target_{int(c)}" for c in ct.columns]
    ct = ct.reset_index().rename(columns={column: column})
    ct["total"] = ct[[c for c in ct.columns if c.startswith("target_")]].sum(axis=1)
    for col in [c for c in ct.columns if c.startswith("target_")]:
        group_total = ct[col].sum()
        ct[f"{col}_pct_within_group"] = (ct[col] / group_total * 100).round(2) if group_total else np.nan
    return ct


def save_group_tables(df: pd.DataFrame) -> None:
    """Persist per-group descriptive tables used by Chapter 1 (theory) and Chapter 3."""
    group_stats = describe_by_group(df, GROUP_STATS_VARIABLES)
    group_stats.to_csv(TABLES_DIR / "heart_disease_group_stats.csv", index=False)

    cp_cross = cross_tab_categorical(df, "cp")
    cp_cross.to_csv(TABLES_DIR / "heart_disease_cp_by_target.csv", index=False)


def save_evidence_index() -> None:
    evidence_rows = [
        {
            "artifact": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "artifact_type": path.parent.name,
            "used_for": "Heart Disease Chapter 2/3 evidence",
        }
        for path in sorted(
            list(FIGURES_DIR.glob("heart_disease_*.png"))
            + list(TABLES_DIR.glob("heart_disease_*.csv"))
            + list(CLEANED_DIR.glob("heart_disease_*.csv"))
            + [LOG_PATH]
        )
    ]
    pd.DataFrame(evidence_rows).to_csv(TABLES_DIR / "heart_disease_evidence_index.csv", index=False)


def main() -> None:
    ensure_dirs()
    setup_logging()
    logging.info("Starting Heart Disease analysis")
    logging.info("Input data path: %s", DATA_PATH.relative_to(ROOT))

    df = load_data()
    cleaned = clean_data(df)
    save_metadata_tables(df, cleaned)
    save_plots(df, cleaned)
    save_group_tables(df)
    save_evidence_index()

    logging.info("Raw shape: %s rows x %s columns", df.shape[0], df.shape[1])
    logging.info("Cleaned shape: %s rows x %s columns", cleaned.shape[0], cleaned.shape[1])
    logging.info("Missing values: %s", df.isna().sum().to_dict())
    logging.info("Outputs written under: %s", OUTPUTS.relative_to(ROOT))
    print(f"Heart Disease analysis complete. Outputs written to {OUTPUTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
