from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "uci_374_appliances_energy_prediction" / "data.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
CLEANED = OUTPUTS / "cleaned"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "appliances_energy_analysis.log"

SELECTED = ["Appliances", "lights", "T1", "RH_1", "T_out", "RH_out", "Windspeed"]


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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d%H:%M:%S", errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, state: str) -> pd.DataFrame:
    rows = []
    for column in SELECTED:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append(
            {
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(method="linear").ffill().bfill()
    for column in SELECTED:
        if column == "Appliances":
            continue
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            cleaned[column] = cleaned[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    cleaned["Appliances_log1p"] = np.log1p(cleaned["Appliances"])
    cleaned["Appliances_diff"] = cleaned["Appliances"].diff()
    return cleaned


def save_tables(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    missing = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(4).values,
        }
    )
    missing.to_csv(TABLES / "appliances_energy_missing_values.csv", index=False)
    pd.concat([summarize(df, "raw"), summarize(cleaned, "cleaned")], ignore_index=True).to_csv(
        TABLES / "appliances_energy_summary.csv", index=False
    )

    time_info = pd.DataFrame(
        [
            {
                "start": df["date"].min(),
                "end": df["date"].max(),
                "samples": len(df),
                "frequency_minutes_mode": df["date"].diff().dt.total_seconds().div(60).mode().iloc[0],
                "target": "Appliances",
            }
        ]
    )
    time_info.to_csv(TABLES / "appliances_energy_time_info.csv", index=False)

    cleaned.to_csv(CLEANED / "appliances_energy_cleaned.csv", index=False)


def save_plots(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    daily = cleaned.set_index("date")["Appliances"].resample("D").mean()
    plt.figure(figsize=(11, 5))
    daily.plot(color="#4C78A8")
    plt.title("Daily mean appliances energy consumption")
    plt.xlabel("Date")
    plt.ylabel("Appliances energy (Wh)")
    # Aligned y-axis shared with the Excel preview (daily mean ranges 37.5-188.54).
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_daily_line.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["Appliances"], bins=40, kde=True, ax=axes[0], color="#F58518")
    axes[0].set_title("Raw Appliances distribution")
    sns.histplot(cleaned["Appliances_log1p"], bins=40, kde=True, ax=axes[1], color="#54A24B")
    axes[1].set_title("Log1p Appliances distribution")
    # Aligned limits shared with the Excel preview so the two versions can be
    # compared on the same axes.
    axes[0].set_xlim(0, 1100)
    axes[0].set_ylim(0, 11000)
    axes[1].set_xlim(2.0, 7.5)
    axes[1].set_ylim(0, 5000)
    fig.tight_layout()
    fig.savefig(FIGURES / "appliances_energy_distribution_raw_vs_log.png", dpi=180)
    plt.close(fig)

    # Phiên bản tách: mỗi histogram một ảnh riêng để dễ trình bày trong báo cáo.
    plt.figure(figsize=(7, 5))
    sns.histplot(df["Appliances"], bins=40, kde=True, color="#F58518")
    plt.title("Raw Appliances distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_distribution_raw.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(cleaned["Appliances_log1p"], bins=40, kde=True, color="#54A24B")
    plt.title("Log1p Appliances distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_distribution_log1p.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=cleaned["Appliances"], color="#B279A2")
    plt.title("Appliances target boxplot")
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_target_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = cleaned[SELECTED].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Appliances Energy selected correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_correlation_heatmap.png", dpi=180)
    plt.close()

    ridgeline = cleaned.copy()
    ridgeline["month"] = ridgeline["date"].dt.strftime("%Y-%m")
    months = sorted(ridgeline["month"].dropna().unique())
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(months))
    # Shared x-axis range with the Excel preview so the two versions align.
    ridgeline_xlim = (2.0, 7.5)
    bins = np.linspace(ridgeline_xlim[0], ridgeline_xlim[1], 80)
    centers = (bins[:-1] + bins[1:]) / 2
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel = kernel / kernel.sum()
    for offset, (month, color) in enumerate(zip(months, palette)):
        values = ridgeline.loc[ridgeline["month"] == month, "Appliances_log1p"].dropna()
        if len(values) < 2:
            continue
        density, _ = np.histogram(values, bins=bins, density=True)
        density = np.convolve(density, kernel, mode="same")
        density = density / density.max() * 0.85 if density.max() > 0 else density
        plt.fill_between(centers, offset, density + offset, color=color, alpha=0.75)
        plt.plot(centers, density + offset, color=color, linewidth=1.0)
    plt.yticks(range(len(months)), months)
    plt.xlabel("log1p(Appliances)")
    plt.ylabel("Month")
    plt.title("Ridgeline plot of Appliances consumption by month")
    plt.xlim(ridgeline_xlim)
    plt.tight_layout()
    plt.savefig(FIGURES / "appliances_energy_monthly_ridgeline.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    setup_logging()
    df = load_data()
    cleaned = clean_data(df)
    save_tables(df, cleaned)
    save_plots(df, cleaned)
    logging.info("Raw shape: %s", df.shape)
    logging.info("Date range: %s to %s", df["date"].min(), df["date"].max())
    print("Appliances Energy analysis complete.")


if __name__ == "__main__":
    main()
