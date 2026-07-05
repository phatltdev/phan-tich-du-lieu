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
APPLIANCES_PATH = ROOT / "outputs" / "cleaned" / "appliances_energy_cleaned.csv"

WORKBOOK_PATH = EXCEL_DIR / "excel_visualization_comparison.xlsx"
SUMMARY_PATH = TABLE_DIR / "excel_visualization_index.csv"
LOG_PATH = LOG_DIR / "create_excel_visualizations.log"


def _ensure_dirs() -> None:
    for path in (EXCEL_DIR, FIGURE_DIR, TABLE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _load_heart() -> pd.DataFrame:
    heart = pd.read_csv(HEART_PATH)
    keep = ["age", "cp", "chol", "thalach", "trestbps", "oldpeak", "target_binary"]
    return heart[keep].copy()


def _load_appliances() -> pd.DataFrame:
    app = pd.read_csv(APPLIANCES_PATH)
    app["date"] = pd.to_datetime(app["date"])
    app["day"] = app["date"].dt.date
    app["Appliances_log1p"] = np.log1p(app["Appliances"])
    app["Appliances_diff"] = app["Appliances"].diff()
    return app


def _create_preview_figures(
    heart: pd.DataFrame,
    cp_freq: pd.DataFrame,
    app_daily: pd.DataFrame,
    app_sample: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "heart_cp_bar": FIGURE_DIR / "excel_heart_cp_bar_preview.png",
        "heart_age_thalach": FIGURE_DIR / "excel_heart_age_thalach_scatter_preview.png",
        "appliances_daily": FIGURE_DIR / "excel_appliances_daily_line_preview.png",
        "appliances_hist": FIGURE_DIR / "excel_appliances_raw_vs_log_hist_preview.png",
    }

    plt.figure(figsize=(7, 4.2))
    plt.bar(cp_freq["cp"].astype(str), cp_freq["count"], color="#4e79a7")
    plt.title("Heart Disease - Chest pain type frequency")
    plt.xlabel("cp")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(paths["heart_cp_bar"], dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    colors = heart["target_binary"].map({0: "#59a14f", 1: "#e15759"})
    plt.scatter(heart["age"], heart["thalach"], c=colors, alpha=0.78, edgecolor="white", linewidth=0.3)
    plt.title("Heart Disease - Age vs maximum heart rate")
    plt.xlabel("Age")
    plt.ylabel("Thalach")
    plt.tight_layout()
    plt.savefig(paths["heart_age_thalach"], dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.2))
    plt.plot(pd.to_datetime(app_daily["day"]), app_daily["mean_appliances"], color="#4e79a7", linewidth=1.5)
    plt.title("Appliances Energy - Daily mean consumption")
    plt.xlabel("Day")
    plt.ylabel("Mean Appliances")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(paths["appliances_daily"], dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.hist(app_sample["Appliances"], bins=30, alpha=0.55, label="Raw", color="#4e79a7")
    plt.hist(app_sample["Appliances_log1p"], bins=30, alpha=0.55, label="log1p", color="#f28e2b")
    plt.title("Appliances Energy - Raw vs log1p distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["appliances_hist"], dpi=160)
    plt.close()

    return paths


def _write_workbook(
    heart: pd.DataFrame,
    cp_freq: pd.DataFrame,
    heart_stats: pd.DataFrame,
    app_daily: pd.DataFrame,
    app_sample: pd.DataFrame,
    app_hist: pd.DataFrame,
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


def main() -> None:
    _ensure_dirs()
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

    previews = _create_preview_figures(heart, cp_freq, app_daily, app_sample)
    _write_workbook(heart, cp_freq, heart_stats, app_daily, app_sample, app_hist)

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
            "chart_type": "Excel column histogram charts",
            "dataset": "Appliances Energy",
            "excel_workbook": "excel/excel_visualization_comparison.xlsx",
            "excel_sheet": "Appliances Histogram",
            "python_source": "src/create_excel_visualizations.py",
            "png_preview": str(previews["appliances_hist"].relative_to(ROOT)).replace("\\", "/"),
            "calculation": "y'_t = ln(1 + y_t)",
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
