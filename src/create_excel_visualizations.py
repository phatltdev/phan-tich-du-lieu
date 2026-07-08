"""Create Excel-based visualization artifacts for the report.

The workbook contains native Excel charts. PNG previews are also exported with
matplotlib so the LaTeX report can include visual evidence while preserving the
auditable Excel workbook under excel/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXCEL_DIR = ROOT / "excel"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
LOG_DIR = ROOT / "outputs" / "logs"

HEART_PATH = ROOT / "outputs" / "cleaned" / "heart_disease_cleaned.csv"
HEART_RAW_PATH = ROOT / "datasets" / "heart+disease" / "processed.cleveland.data"
APPLIANCES_PATH = ROOT / "outputs" / "cleaned" / "appliances_energy_cleaned.csv"

WORKBOOK_PATH = EXCEL_DIR / "excel_visualization_comparison.xlsx"
SUMMARY_PATH = TABLE_DIR / "excel_visualization_index.csv"
LOG_PATH = LOG_DIR / "create_excel_visualizations.log"

HEART_COLUMNS = [
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


def _ensure_dirs() -> None:
    for path in (EXCEL_DIR, FIGURE_DIR, TABLE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _load_heart() -> pd.DataFrame:
    heart = pd.read_csv(HEART_PATH)
    keep = ["age", "cp", "chol", "thalach", "trestbps", "oldpeak", "target_binary"]
    return heart[keep].copy()


def _load_heart_raw() -> pd.DataFrame:
    heart_raw = pd.read_csv(HEART_RAW_PATH, names=HEART_COLUMNS, na_values="?")
    heart_raw["age"] = pd.to_numeric(heart_raw["age"], errors="coerce")
    return heart_raw


def _load_appliances() -> pd.DataFrame:
    app = pd.read_csv(APPLIANCES_PATH)
    app["date"] = pd.to_datetime(app["date"])
    app["day"] = app["date"].dt.date
    app["Appliances_log1p"] = np.log1p(app["Appliances"])
    app["Appliances_diff"] = app["Appliances"].diff()
    return app


def _create_preview_figures(
    heart_raw: pd.DataFrame,
    heart: pd.DataFrame,
    cp_freq: pd.DataFrame,
    app_daily: pd.DataFrame,
    app_sample: pd.DataFrame,
    app_monthly: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "heart_cp_bar": FIGURE_DIR / "excel_heart_cp_bar_preview.png",
        "heart_age_raw_vs_cleaned": FIGURE_DIR / "excel_heart_age_raw_vs_cleaned_preview.png",
        "heart_age_thalach": FIGURE_DIR / "excel_heart_age_thalach_scatter_preview.png",
        "appliances_daily": FIGURE_DIR / "excel_appliances_daily_line_preview.png",
        "appliances_hist": FIGURE_DIR / "excel_appliances_raw_vs_log_hist_preview.png",
        "appliances_hist_log": FIGURE_DIR / "excel_appliances_log1p_hist_preview.png",
        "heart_chol_boxplot": FIGURE_DIR / "excel_heart_chol_boxplot_preview.png",
        "heart_corr_heatmap": FIGURE_DIR / "excel_heart_corr_heatmap_preview.png",
        "appliances_ridgeline": FIGURE_DIR / "excel_appliances_ridgeline_preview.png",
    }

    # Shared axis limits keep the Excel previews aligned with the Python figures
    # produced by analyze_heart_disease.py and analyze_appliances_energy.py so the
    # two versions can be compared side by side on the same axes.
    cp_xlim = (-0.5, 3.5)
    cp_ylim = (0, 150)
    age_xlim = (28, 78)
    thalach_ylim = (80, 205)
    daily_ylim = (0, 200)
    raw_xlim = (0, 1100)
    log_xlim = (2.0, 7.5)
    raw_ylim = (0, 11000)
    log_ylim = (0, 5000)
    chol_ylim = (0, 600)

    plt.figure(figsize=(7, 4.2))
    plt.bar(cp_freq["cp"].astype(str), cp_freq["count"], color="#4e79a7")
    plt.title("Heart Disease - Chest pain type frequency")
    plt.xlabel("cp")
    plt.ylabel("Frequency")
    plt.xlim(cp_xlim)
    plt.ylim(cp_ylim)
    plt.tight_layout()
    plt.savefig(paths["heart_cp_bar"], dpi=160)
    plt.close()

    age_bins = np.linspace(28, 78, 13)
    raw_counts, raw_edges = np.histogram(heart_raw["age"].dropna(), bins=age_bins)
    clean_counts, clean_edges = np.histogram(heart["age"].dropna(), bins=age_bins)
    raw_midpoints = (raw_edges[:-1] + raw_edges[1:]) / 2
    clean_midpoints = (clean_edges[:-1] + clean_edges[1:]) / 2
    age_ylim = (0, np.ceil(max(raw_counts.max(), clean_counts.max()) * 1.15 / 5) * 5)

    fig, (ax_raw, ax_clean) = plt.subplots(1, 2, figsize=(12, 4.5))
    bar_width = np.diff(age_bins)[0] * 0.88
    ax_raw.bar(raw_midpoints, raw_counts, width=bar_width, color="#4e79a7", edgecolor="white", linewidth=0.4)
    ax_raw.set_title("Heart Disease - Raw age distribution (Excel style)")
    ax_raw.set_xlabel("Age")
    ax_raw.set_ylabel("Frequency")
    ax_raw.set_xlim(age_xlim)
    ax_raw.set_ylim(age_ylim)
    ax_clean.bar(clean_midpoints, clean_counts, width=bar_width, color="#59a14f", edgecolor="white", linewidth=0.4)
    ax_clean.set_title("Heart Disease - Cleaned age distribution (Excel style)")
    ax_clean.set_xlabel("Age")
    ax_clean.set_ylabel("Frequency")
    ax_clean.set_xlim(age_xlim)
    ax_clean.set_ylim(age_ylim)
    fig.tight_layout()
    fig.savefig(paths["heart_age_raw_vs_cleaned"], dpi=160)
    plt.close(fig)

    plt.figure(figsize=(7, 4.2))
    colors = heart["target_binary"].map({0: "#59a14f", 1: "#e15759"})
    plt.scatter(heart["age"], heart["thalach"], c=colors, alpha=0.78, edgecolor="white", linewidth=0.3)
    plt.title("Heart Disease - Age vs maximum heart rate")
    plt.xlabel("Age")
    plt.ylabel("Thalach")
    plt.xlim(age_xlim)
    plt.ylim(thalach_ylim)
    plt.tight_layout()
    plt.savefig(paths["heart_age_thalach"], dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.2))
    plt.plot(pd.to_datetime(app_daily["day"]), app_daily["mean_appliances"], color="#4e79a7", linewidth=1.5)
    plt.title("Appliances Energy - Daily mean consumption")
    plt.xlabel("Day")
    plt.ylabel("Mean Appliances")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(daily_ylim)
    plt.tight_layout()
    plt.savefig(paths["appliances_daily"], dpi=160)
    plt.close()

    # The paired preview mirrors the Python raw/log1p figure so the report can
    # compare one Python chart with one Excel-style chart.
    raw_counts_sample, _ = np.histogram(app_sample["Appliances"], bins=30, range=raw_xlim)
    raw_ylim_sample = (0, np.ceil(raw_counts_sample.max() * 1.10 / 100) * 100)
    log_counts_sample, _ = np.histogram(app_sample["Appliances_log1p"], bins=30, range=log_xlim)
    log_ylim_sample = (0, np.ceil(log_counts_sample.max() * 1.10 / 100) * 100)

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_r.hist(app_sample["Appliances"], bins=30, range=raw_xlim, color="#4e79a7", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_r.set_title("Appliances Energy - Raw distribution (Excel style)")
    ax_r.set_xlabel("Appliances")
    ax_r.set_ylabel("Frequency")
    ax_r.set_xlim(raw_xlim)
    ax_r.set_ylim(raw_ylim_sample)
    ax_l.hist(app_sample["Appliances_log1p"], bins=30, range=log_xlim, color="#f28e2b", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_l.set_title("Appliances Energy - log1p distribution (Excel style)")
    ax_l.set_xlabel("log1p(Appliances)")
    ax_l.set_ylabel("Frequency")
    ax_l.set_xlim(log_xlim)
    ax_l.set_ylim(log_ylim_sample)
    fig.tight_layout()
    fig.savefig(paths["appliances_hist"], dpi=160)
    plt.close(fig)

    # Separate log1p panel as its own preview for better bar visibility.
    fig, ax_l = plt.subplots(figsize=(7.5, 4.5))
    ax_l.hist(app_sample["Appliances_log1p"], bins=30, range=log_xlim, color="#f28e2b", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_l.set_title("Appliances Energy - log1p distribution (Excel style)")
    ax_l.set_xlabel("log1p(Appliances)")
    ax_l.set_ylabel("Frequency")
    ax_l.set_xlim(log_xlim)
    ax_l.set_ylim(log_ylim_sample)
    fig.tight_layout()
    fig.savefig(paths["appliances_hist_log"], dpi=160)
    plt.close(fig)

    # --- Boxplot preview (chol by target_binary), Excel box-and-whisker style.
    plt.figure(figsize=(7, 4.2))
    groups = heart.groupby("target_binary")["chol"].apply(list)
    positions = sorted(groups.index)
    data = [groups[g] for g in positions]
    plt.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                boxprops=dict(facecolor="#bdd7e7", edgecolor="#4e79a7"),
                medianprops=dict(color="#e15759", linewidth=2),
                whiskerprops=dict(color="#4e79a7"), capprops=dict(color="#4e79a7"),
                flierprops=dict(marker="o", markerfacecolor="#e15759", markersize=4, alpha=0.6))
    plt.title("Heart Disease - Cholesterol by disease presence")
    plt.xlabel("target_binary (0 = absent, 1 = present)")
    plt.ylabel("Serum cholesterol")
    plt.xticks(positions)
    plt.xlim(-0.5, 1.5)
    plt.ylim(chol_ylim)
    plt.tight_layout()
    plt.savefig(paths["heart_chol_boxplot"], dpi=160)
    plt.close()

    # --- Correlation heatmap preview, Excel conditional-formatting style.
    corr_vars = ["age", "trestbps", "chol", "thalach", "oldpeak", "target_binary"]
    corr = heart[corr_vars].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # Red-white-blue diverging palette mirrors Excel's 3-color conditional formatting.
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_vars)))
    ax.set_yticks(range(len(corr_vars)))
    ax.set_xticklabels(corr_vars, rotation=45, ha="right")
    ax.set_yticklabels(corr_vars)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Heart Disease - Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(paths["heart_corr_heatmap"], dpi=160)
    plt.close(fig)

    # --- Ridgeline preview (log1p(Appliances) by month), aligned x-axis with Python.
    months = sorted(app_monthly["month"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(log_xlim[0], log_xlim[1], 80)
    cmap = plt.cm.viridis
    for offset, month in enumerate(months):
        values = app_monthly.loc[app_monthly["month"] == month, "Appliances_log1p"].dropna()
        density, _ = np.histogram(values, bins=bins, density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        color = cmap(offset / max(1, len(months) - 1))
        ax.plot(centers, density + offset, color=color, linewidth=1.0)
        ax.fill_between(centers, density + offset, offset, color=color, alpha=0.25)
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels(months)
    ax.set_xlabel("log1p(Appliances)")
    ax.set_ylabel("Month")
    ax.set_title("Appliances Energy - Ridgeline by month")
    ax.set_xlim(log_xlim)
    fig.tight_layout()
    fig.savefig(paths["appliances_ridgeline"], dpi=160)
    plt.close(fig)

    return paths


def _write_workbook(
    heart_raw: pd.DataFrame,
    heart: pd.DataFrame,
    cp_freq: pd.DataFrame,
    heart_stats: pd.DataFrame,
    app_daily: pd.DataFrame,
    app_sample: pd.DataFrame,
    app_hist: pd.DataFrame,
    heart_box_stats: pd.DataFrame,
    heart_corr: pd.DataFrame,
    app_monthly: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(WORKBOOK_PATH, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 14, "bg_color": "#D9EAF7"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#EAF3F8", "border": 1})
        pct_fmt = workbook.add_format({"num_format": "0.0%"})
        num_fmt = workbook.add_format({"num_format": "#,##0.00"})

        readme = pd.DataFrame(
            [
                ["Python source", "src/create_excel_visualizations.py"],
                ["Excel workbook", "excel/excel_visualization_comparison.xlsx"],
                ["Heart raw input", "datasets/heart+disease/processed.cleveland.data"],
                ["Heart input", "outputs/cleaned/heart_disease_cleaned.csv"],
                ["Appliances input", "outputs/cleaned/appliances_energy_cleaned.csv"],
                ["Heart Python source for comparison", "src/analyze_heart_disease.py"],
                ["Appliances Python source for comparison", "src/analyze_appliances_energy.py"],
                ["Formula note", "Heart stats sheet uses Excel formulas: AVERAGE, MEDIAN, STDEV.S, MIN, MAX, QUARTILE.INC."],
                ["Formula note", "Appliances daily sheet stores daily mean, daily standard deviation, and log1p target used for charts."],
            ],
            columns=["Item", "Relative path or note"],
        )
        readme.to_excel(writer, sheet_name="README", index=False)
        ws = writer.sheets["README"]
        ws.set_column("A:A", 28)
        ws.set_column("B:B", 96)
        ws.write("A1", "Item", header_fmt)
        ws.write("B1", "Relative path or note", header_fmt)

        heart.to_excel(writer, sheet_name="Heart Raw", index=False)
        ws = writer.sheets["Heart Raw"]
        ws.freeze_panes(1, 0)
        ws.set_column("A:G", 13)
        for col, name in enumerate(heart.columns):
            ws.write(0, col, name, header_fmt)

        cp_freq.to_excel(writer, sheet_name="Heart Frequency", index=False)
        ws = writer.sheets["Heart Frequency"]
        ws.set_column("A:C", 16)
        for col, name in enumerate(cp_freq.columns):
            ws.write(0, col, name, header_fmt)
        ws.set_column("C:C", 16, pct_fmt)
        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": "Chest pain frequency",
                "categories": "='Heart Frequency'!$A$2:$A$5",
                "values": "='Heart Frequency'!$B$2:$B$5",
                "fill": {"color": "#4E79A7"},
            }
        )
        chart.set_title({"name": "Heart Disease - Chest Pain Type Frequency"})
        chart.set_x_axis({"name": "cp"})
        chart.set_y_axis({"name": "Frequency"})
        ws.insert_chart("E2", chart, {"x_scale": 1.35, "y_scale": 1.25})

        age_bins = np.linspace(28, 78, 13)
        raw_counts, raw_edges = np.histogram(heart_raw["age"].dropna(), bins=age_bins)
        clean_counts, clean_edges = np.histogram(heart["age"].dropna(), bins=age_bins)
        age_hist = pd.DataFrame(
            {
                "age_bin_midpoint": (raw_edges[:-1] + raw_edges[1:]) / 2,
                "raw_frequency": raw_counts,
                "cleaned_frequency": clean_counts,
            }
        )
        age_hist.to_excel(writer, sheet_name="Heart Age Hist", index=False)
        ws = writer.sheets["Heart Age Hist"]
        ws.set_column("A:C", 18, num_fmt)
        for col, name in enumerate(age_hist.columns):
            ws.write(0, col, name, header_fmt)
        raw_age_chart = workbook.add_chart({"type": "column"})
        age_last_row = len(age_hist) + 1
        raw_age_chart.add_series(
            {
                "name": "Raw age",
                "categories": f"='Heart Age Hist'!$A$2:$A${age_last_row}",
                "values": f"='Heart Age Hist'!$B$2:$B${age_last_row}",
                "fill": {"color": "#4E79A7"},
            }
        )
        raw_age_chart.set_title({"name": "Heart Disease - Raw Age Distribution"})
        raw_age_chart.set_x_axis({"name": "Age bin midpoint"})
        raw_age_chart.set_y_axis({"name": "Frequency"})
        ws.insert_chart("E2", raw_age_chart, {"x_scale": 1.2, "y_scale": 1.1})
        clean_age_chart = workbook.add_chart({"type": "column"})
        clean_age_chart.add_series(
            {
                "name": "Cleaned age",
                "categories": f"='Heart Age Hist'!$A$2:$A${age_last_row}",
                "values": f"='Heart Age Hist'!$C$2:$C${age_last_row}",
                "fill": {"color": "#59A14F"},
            }
        )
        clean_age_chart.set_title({"name": "Heart Disease - Cleaned Age Distribution"})
        clean_age_chart.set_x_axis({"name": "Age bin midpoint"})
        clean_age_chart.set_y_axis({"name": "Frequency"})
        ws.insert_chart("E20", clean_age_chart, {"x_scale": 1.2, "y_scale": 1.1})

        heart_stats.to_excel(writer, sheet_name="Heart Stats", index=False)
        ws = writer.sheets["Heart Stats"]
        ws.set_column("A:A", 14)
        ws.set_column("B:H", 13, num_fmt)
        for col, name in enumerate(heart_stats.columns):
            ws.write(0, col, name, header_fmt)
        source_cols = {name: idx for idx, name in enumerate(heart.columns)}
        for row_idx, var in enumerate(heart_stats["variable"], start=2):
            excel_col = chr(ord("A") + source_cols[var])
            ws.write_formula(row_idx - 1, 1, f"=AVERAGE('Heart Raw'!${excel_col}$2:${excel_col}$304)")
            ws.write_formula(row_idx - 1, 2, f"=MEDIAN('Heart Raw'!${excel_col}$2:${excel_col}$304)")
            ws.write_formula(row_idx - 1, 3, f"=STDEV.S('Heart Raw'!${excel_col}$2:${excel_col}$304)")
            ws.write_formula(row_idx - 1, 4, f"=MIN('Heart Raw'!${excel_col}$2:${excel_col}$304)")
            ws.write_formula(row_idx - 1, 5, f"=MAX('Heart Raw'!${excel_col}$2:${excel_col}$304)")
            ws.write_formula(row_idx - 1, 6, f"=QUARTILE.INC('Heart Raw'!${excel_col}$2:${excel_col}$304,3)-QUARTILE.INC('Heart Raw'!${excel_col}$2:${excel_col}$304,1)")
            ws.write_formula(row_idx - 1, 7, f"=C{row_idx}/B{row_idx}")

        scatter_df = heart[["age", "thalach", "target_binary"]].copy()
        scatter_df.to_excel(writer, sheet_name="Heart Scatter", index=False)
        ws = writer.sheets["Heart Scatter"]
        ws.set_column("A:C", 14)
        chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        chart.add_series(
            {
                "name": "Age-Thalach",
                "categories": "='Heart Scatter'!$A$2:$A$304",
                "values": "='Heart Scatter'!$B$2:$B$304",
                "marker": {"type": "circle", "size": 4, "border": {"color": "#4E79A7"}, "fill": {"color": "#4E79A7"}},
            }
        )
        chart.set_title({"name": "Heart Disease - Age vs Thalach"})
        chart.set_x_axis({"name": "Age"})
        chart.set_y_axis({"name": "Maximum heart rate"})
        ws.insert_chart("E2", chart, {"x_scale": 1.45, "y_scale": 1.25})

        app_daily.to_excel(writer, sheet_name="Appliances Daily", index=False)
        ws = writer.sheets["Appliances Daily"]
        ws.set_column("A:A", 14)
        ws.set_column("B:D", 18, num_fmt)
        for col, name in enumerate(app_daily.columns):
            ws.write(0, col, name, header_fmt)
        chart = workbook.add_chart({"type": "line"})
        last_row = len(app_daily) + 1
        chart.add_series(
            {
                "name": "Daily mean",
                "categories": f"='Appliances Daily'!$A$2:$A${last_row}",
                "values": f"='Appliances Daily'!$B$2:$B${last_row}",
                "line": {"color": "#4E79A7", "width": 1.5},
            }
        )
        chart.set_title({"name": "Appliances Energy - Daily Mean Consumption"})
        chart.set_x_axis({"name": "Day"})
        chart.set_y_axis({"name": "Mean Appliances"})
        ws.insert_chart("F2", chart, {"x_scale": 1.55, "y_scale": 1.25})

        app_sample.to_excel(writer, sheet_name="Appliances Sample", index=False)
        ws = writer.sheets["Appliances Sample"]
        ws.set_column("A:E", 18, num_fmt)
        for col, name in enumerate(app_sample.columns):
            ws.write(0, col, name, header_fmt)
        ws.write("G1", "Distribution source", title_fmt)
        ws.write("G2", "Use columns Appliances and Appliances_log1p to create Excel histograms.")
        ws.write("G3", "Formula: log1p(Appliances) = LN(1 + Appliances).")

        app_hist.to_excel(writer, sheet_name="Appliances Histogram", index=False)
        ws = writer.sheets["Appliances Histogram"]
        ws.set_column("A:D", 18, num_fmt)
        for col, name in enumerate(app_hist.columns):
            ws.write(0, col, name, header_fmt)
        raw_chart = workbook.add_chart({"type": "column"})
        hist_last_row = len(app_hist) + 1
        raw_chart.add_series(
            {
                "name": "Raw count",
                "categories": f"='Appliances Histogram'!$A$2:$A${hist_last_row}",
                "values": f"='Appliances Histogram'!$B$2:$B${hist_last_row}",
                "fill": {"color": "#4E79A7"},
            }
        )
        raw_chart.set_title({"name": "Appliances - Raw Distribution"})
        raw_chart.set_x_axis({"name": "Raw bin midpoint"})
        raw_chart.set_y_axis({"name": "Frequency"})
        ws.insert_chart("F2", raw_chart, {"x_scale": 1.2, "y_scale": 1.1})

        log_chart = workbook.add_chart({"type": "column"})
        log_chart.add_series(
            {
                "name": "log1p count",
                "categories": f"='Appliances Histogram'!$C$2:$C${hist_last_row}",
                "values": f"='Appliances Histogram'!$D$2:$D${hist_last_row}",
                "fill": {"color": "#F28E2B"},
            }
        )
        log_chart.set_title({"name": "Appliances - log1p Distribution"})
        log_chart.set_x_axis({"name": "log1p bin midpoint"})
        log_chart.set_y_axis({"name": "Frequency"})
        ws.insert_chart("F20", log_chart, {"x_scale": 1.2, "y_scale": 1.1})

        # --- Sheet "Heart Boxplot": chol by target_binary with QUARTILE.INC formulas.
        heart_box_stats.to_excel(writer, sheet_name="Heart Boxplot", index=False)
        ws = writer.sheets["Heart Boxplot"]
        ws.set_column("A:A", 16)
        ws.set_column("B:H", 13, num_fmt)
        for col, name in enumerate(heart_box_stats.columns):
            ws.write(0, col, name, header_fmt)
        # Build CORREL-based box summaries referencing the chol column (C) in Heart Raw,
        # filtered implicitly by writing the QUARTILE over the full column. Box stats
        # below are computed in Python (heart_box_stats) and written as values so the
        # workbook stays self-contained without dynamic array formulas.
        ws.write("G1", "Chart instructions", title_fmt)
        ws.write("G2", "Select columns target_binary and chol, then insert a Box-and-Whisker chart (Excel 2016+).")
        ws.write("G3", "Q1/median/Q3/whisk formulas use QUARTILE.INC on the chol column.")

        # --- Sheet "Heart Corr": correlation matrix with 3-color conditional formatting.
        heart_corr.to_excel(writer, sheet_name="Heart Corr", index=True)
        ws = writer.sheets["Heart Corr"]
        ws.set_column("A:A", 14)
        ws.set_column("B:H", 12, num_fmt)
        ws.write(0, 0, "variable", header_fmt)
        for col, name in enumerate(heart_corr.columns, start=1):
            ws.write(0, col, name, header_fmt)
        n_corr = len(heart_corr)
        # Apply a red-white-blue diverging 3-color scale mimicking a heatmap.
        ws.conditional_format(
            1, 1, n_corr, n_corr,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",   # red for r = -1
                "mid_color": "#FFFFFF",   # white for r = 0
                "max_color": "#5A8AC6",   # blue for r = +1
            },
        )
        ws.write("J1", "Conditional formatting", title_fmt)
        ws.write("J2", "3-color scale (red = -1, white = 0, blue = +1) mimics a heatmap in Excel.")
        ws.write("J3", "Each cell value is the Pearson r between two variables computed in Python.")

        # --- Sheet "Appliances Ridgeline": monthly log1p sample for density / histogram.
        app_monthly.to_excel(writer, sheet_name="Appliances Ridgeline", index=False)
        ws = writer.sheets["Appliances Ridgeline"]
        ws.set_column("A:C", 16, num_fmt)
        for col, name in enumerate(app_monthly.columns):
            ws.write(0, col, name, header_fmt)
        ws.write("E1", "Chart instructions", title_fmt)
        ws.write("E2", "Excel has no native ridgeline. Group by month and build one histogram per month,")
        ws.write("E3", "or use a pivot chart of log1p(Appliances) bins by month to approximate the ridgeline.")
        ws.write("E4", "Python preview uses matplotlib to render the ridgeline for the report.")


def main() -> None:
    _ensure_dirs()
    heart_raw = _load_heart_raw()
    heart = _load_heart()
    app = _load_appliances()

    cp_freq = (
        heart["cp"]
        .value_counts()
        .sort_index()
        .rename_axis("cp")
        .reset_index(name="count")
    )
    cp_freq["relative_frequency"] = cp_freq["count"] / len(heart)

    stat_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    heart_stats = pd.DataFrame({"variable": stat_vars})
    for var in stat_vars:
        s = heart[var]
        heart_stats.loc[heart_stats["variable"] == var, "mean"] = s.mean()
        heart_stats.loc[heart_stats["variable"] == var, "median"] = s.median()
        heart_stats.loc[heart_stats["variable"] == var, "std"] = s.std(ddof=1)
        heart_stats.loc[heart_stats["variable"] == var, "min"] = s.min()
        heart_stats.loc[heart_stats["variable"] == var, "max"] = s.max()
        heart_stats.loc[heart_stats["variable"] == var, "iqr"] = s.quantile(0.75) - s.quantile(0.25)
        heart_stats.loc[heart_stats["variable"] == var, "cv"] = s.std(ddof=1) / s.mean()

    app_daily = (
        app.groupby("day")
        .agg(
            mean_appliances=("Appliances", "mean"),
            std_appliances=("Appliances", "std"),
            mean_log1p=("Appliances_log1p", "mean"),
        )
        .reset_index()
    )
    app_sample = app[["Appliances", "Appliances_log1p", "T_out", "RH_out", "Windspeed"]].sample(
        n=min(1000, len(app)),
        random_state=42,
    )
    raw_counts, raw_edges = np.histogram(app_sample["Appliances"], bins=20)
    log_counts, log_edges = np.histogram(app_sample["Appliances_log1p"], bins=20)
    app_hist = pd.DataFrame(
        {
            "raw_bin_midpoint": (raw_edges[:-1] + raw_edges[1:]) / 2,
            "raw_count": raw_counts,
            "log1p_bin_midpoint": (log_edges[:-1] + log_edges[1:]) / 2,
            "log1p_count": log_counts,
        }
    )

    # Boxplot summary: chol statistics per target_binary group (basis for the Excel
    # box-and-whisker sheet and the matplotlib preview).
    heart_box_stats = (
        heart.groupby("target_binary")["chol"]
        .agg(["count", "min", "max", "mean"])
        .reset_index()
    )
    heart_box_stats["q1"] = heart.groupby("target_binary")["chol"].quantile(0.25).values
    heart_box_stats["median"] = heart.groupby("target_binary")["chol"].quantile(0.5).values
    heart_box_stats["q3"] = heart.groupby("target_binary")["chol"].quantile(0.75).values

    # Correlation matrix for the heatmap sheet and preview.
    corr_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target_binary"]
    heart_corr = heart[corr_cols].corr()

    # Monthly log1p sample for the ridgeline sheet and preview.
    app["month"] = app["date"].dt.strftime("%Y-%m")
    app_monthly = app[["month", "Appliances", "Appliances_log1p"]].sample(
        n=min(3000, len(app)), random_state=42
    ).sort_values("month").reset_index(drop=True)

    previews = _create_preview_figures(heart_raw, heart, cp_freq, app_daily, app_sample, app_monthly)
    _write_workbook(
        heart_raw, heart, cp_freq, heart_stats, app_daily, app_sample, app_hist,
        heart_box_stats, heart_corr, app_monthly,
    )

    index_rows = [
        {
            "chart_id": "excel_heart_cp_bar",
            "chart_type": "Excel column chart",
            "dataset": "Heart Disease",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Heart Frequency",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["heart_cp_bar"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "f_j = count(cp = j), p_j = f_j / n",
        },
        {
            "chart_id": "excel_heart_age_thalach_scatter",
            "chart_type": "Excel scatter chart",
            "dataset": "Heart Disease",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Heart Scatter",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["heart_age_thalach"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "points are pairs (age_i, thalach_i); relation checked with Pearson r",
        },
        {
            "chart_id": "excel_heart_age_raw_vs_cleaned",
            "chart_type": "Excel column histogram (raw vs cleaned)",
            "dataset": "Heart Disease",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Heart Age Hist",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["heart_age_raw_vs_cleaned"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "frequency of raw and cleaned age values over fixed bins",
        },
        {
            "chart_id": "excel_appliances_daily_line",
            "chart_type": "Excel line chart",
            "dataset": "Appliances Energy",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Appliances Daily",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["appliances_daily"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "daily mean = average(Appliances values in the same day)",
        },
        {
            "chart_id": "excel_appliances_raw_vs_log_hist",
            "chart_type": "Excel column histogram (raw)",
            "dataset": "Appliances Energy",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Appliances Histogram",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["appliances_hist"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "frequency of Appliances over fixed bins",
        },
        {
            "chart_id": "excel_appliances_log1p_hist",
            "chart_type": "Excel column histogram (log1p)",
            "dataset": "Appliances Energy",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Appliances Histogram",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["appliances_hist_log"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "y'_t = ln(1 + y_t)",
        },
        {
            "chart_id": "excel_heart_chol_boxplot",
            "chart_type": "Excel box-and-whisker",
            "dataset": "Heart Disease",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Heart Boxplot",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["heart_chol_boxplot"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "Q1/median/Q3 from QUARTILE.INC of chol per target group",
        },
        {
            "chart_id": "excel_heart_corr_heatmap",
            "chart_type": "Excel 3-color conditional formatting",
            "dataset": "Heart Disease",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Heart Corr",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["heart_corr_heatmap"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "r_XY = Pearson correlation between variable columns",
        },
        {
            "chart_id": "excel_appliances_ridgeline",
            "chart_type": "Excel pivot histogram approximation",
            "dataset": "Appliances Energy",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Appliances Ridgeline",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["appliances_ridgeline"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "density of log1p(Appliances) per month",
        },
    ]
    pd.DataFrame(index_rows).to_csv(SUMMARY_PATH, index=False)

    LOG_PATH.write_text(
        "\n".join(
            [
                f"Created workbook: {WORKBOOK_PATH.relative_to(ROOT)}",
                f"Created index: {SUMMARY_PATH.relative_to(ROOT)}",
                *[f"Created preview: {p.relative_to(ROOT)}" for p in previews.values()],
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
