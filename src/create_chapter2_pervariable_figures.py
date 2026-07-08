"""Generate per-variable histogram + boxplot figures for Chương 2 (datasets).

Mirrors the per-variable analysis style of the reference report
(Nhom4_v2.pdf §3.1.3): for a selected set of features in each dataset, draw a
histogram (whole / by target group) and a boxplot split by the target/group
variable. All numbers come from real cleaned CSVs already in the project;
nothing is fabricated.

Outputs PNG files named ``ch2_pervar_<dataset>_<var>.png`` into
``outputs/figures/``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
CLEANED = ROOT / "outputs" / "cleaned"
HAR_BASE = ROOT / "datasets" / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
MHEALTH_DIR = ROOT / "datasets" / "MHEALTHDATASET" / "MHEALTHDATASET"

GREEN = "#2ca02c"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GRAY = "#7f7f7f"


def save_pair(fig_hist: plt.Figure, fig_box: plt.Figure, dataset: str, var: str) -> None:
    stem = f"ch2_pervar_{dataset}_{var}"
    fig_hist.savefig(FIG_DIR / f"{stem}_hist.png", dpi=120, bbox_inches="tight")
    fig_box.savefig(FIG_DIR / f"{stem}_box.png", dpi=120, bbox_inches="tight")
    plt.close(fig_hist)
    plt.close(fig_box)


def numeric_hist_box(df: pd.DataFrame, var: str, group_col: str, dataset: str,
                     var_label: str, group_labels: dict | None = None,
                     bins: int = 25) -> None:
    """Histogram (3 panels: whole / group A / group B) + boxplot by group."""
    data = df[[var, group_col]].dropna()
    if data.empty:
        return
    groups = sorted(data[group_col].dropna().unique())
    # For binary targets keep a stable [0,1] order; cap at first 2 groups for the
    # histogram split so the layout matches the reference (whole / yes / no).
    if len(groups) > 2:
        groups = groups[:2]

    fig_h, axes = plt.subplots(1, len(groups) + 1, figsize=(4 * (len(groups) + 1), 3.2))
    if len(groups) + 1 == 1:
        axes = [axes]
    axes[0].hist(data[var], bins=bins, color=BLUE, alpha=0.8, edgecolor="white")
    axes[0].set_title(f"Toàn bộ: {var_label}", fontsize=10)
    axes[0].set_xlabel(var_label)
    axes[0].set_ylabel("Tần suất")
    palette = [ORANGE, GREEN, RED, GRAY]
    for i, g in enumerate(groups):
        sub = data[data[group_col] == g][var]
        lbl = group_labels.get(g, str(g)) if group_labels else str(g)
        axes[i + 1].hist(sub, bins=bins, color=palette[i % len(palette)], alpha=0.8, edgecolor="white")
        axes[i + 1].set_title(f"{lbl}: {var_label}", fontsize=10)
        axes[i + 1].set_xlabel(var_label)
    fig_h.tight_layout()

    fig_b, axb = plt.subplots(figsize=(4.5, 3.4))
    plot_groups = []
    plot_labels = []
    for g in groups:
        sub = data[data[group_col] == g][var]
        if len(sub) > 0:
            plot_groups.append(sub.values)
            plot_labels.append(group_labels.get(g, str(g)) if group_labels else str(g))
    if plot_groups:
        bp = axb.boxplot(plot_groups, labels=plot_labels, patch_artist=True, showmeans=True,
                         meanprops={"marker": "x", "markeredgecolor": "black", "markersize": 6})
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    axb.set_ylabel(var_label)
    axb.set_title(f"Box plot {var_label} theo nhóm", fontsize=10)
    fig_b.tight_layout()

    save_pair(fig_h, fig_b, dataset, var)


def categorical_bar(df: pd.DataFrame, var: str, group_col: str, dataset: str,
                    var_label: str) -> None:
    """Grouped bar chart of category counts split by target group."""
    data = df[[var, group_col]].dropna()
    if data.empty:
        return
    ct = pd.crosstab(data[var], data[group_col])
    cats = ct.index.tolist()
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(cats) + 2), 3.6))
    x = np.arange(len(cats))
    width = 0.8 / max(1, ct.shape[1])
    palette = [ORANGE, GREEN, RED, GRAY, BLUE]
    for j, col in enumerate(ct.columns):
        ax.bar(x + j * width - 0.4 + width / 2, ct[col].values, width,
               label=str(col), color=palette[j % len(palette)], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats], rotation=0)
    ax.set_xlabel(var_label)
    ax.set_ylabel("Số lượng")
    ax.set_title(f"Tần suất {var_label} theo {group_col}", fontsize=10)
    ax.legend(title=group_col, fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / f"ch2_pervar_{dataset}_{var}_bar.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Group A: tabular
# --------------------------------------------------------------------------- #
def group_a_heart() -> None:
    df = pd.read_csv(CLEANED / "heart_disease_cleaned.csv")
    df["has_disease"] = (df["num"] > 0).astype(int)  # 0 = healthy, 1 = disease
    labels = {0: "Không bệnh", 1: "Có bệnh"}
    for var, lbl in [("age", "Tuổi"), ("trestbps", "Huyết áp nghỉ"),
                     ("chol", "Cholesterol"), ("thalach", "Nhịp tim tối đa"),
                     ("oldpeak", "ST depression")]:
        numeric_hist_box(df, var, "has_disease", "heart", lbl, labels)
    for var, lbl in [("cp", "Loại đau ngực"), ("sex", "Giới tính"),
                     ("restecg", "ECG nghỉ"), ("slope", "Độ dốc ST")]:
        categorical_bar(df, var, "has_disease", "heart", lbl)


def group_a_cdc() -> None:
    df = pd.read_csv(CLEANED / "cdc_diabetes_cleaned.csv")
    target = "Diabetes_binary"
    if target not in df.columns:
        # fall back to the joined column name used in some builds
        for c in df.columns:
            if "Diabetes" in c:
                target = c
                break
    labels = {0: "Không tiểu đường", 1: "Tiểu đường"}
    for var, lbl in [("BMI", "BMI"), ("MentHlth", "Ngày sức khỏe tinh thần"),
                     ("PhysHlth", "Ngày sức khỏe thể chất"), ("Age", "Nhóm tuổi"),
                     ("GenHlth", "Sức khỏe tổng quát")]:
        numeric_hist_box(df, var, target, "cdc", lbl, labels)
    for var, lbl in [("HighBP", "Huyết áp cao"), ("HighChol", "Cholesterol cao"),
                     ("Smoker", "Hút thuốc"), ("PhysActivity", "Hoạt động thể chất")]:
        categorical_bar(df, var, target, "cdc", lbl)


def group_a_breast() -> None:
    df = pd.read_csv(CLEANED / "breast_cancer_cleaned.csv")
    diag_col = "Diagnosis" if "Diagnosis" in df.columns else df.columns[-1]
    labels = {"B": "Lành tính (B)", "M": "Ác tính (M)"}
    for var, lbl in [("radius1", "Bán kính"), ("texture1", "Kết cấu"),
                     ("perimeter1", "Chu vi"), ("area1", "Diện tích"),
                     ("concavity1", "Độ lõm")]:
        numeric_hist_box(df, var, diag_col, "breast", lbl, labels)


# --------------------------------------------------------------------------- #
# Group B: time series / sensors
# --------------------------------------------------------------------------- #
def group_b_appliances() -> None:
    df = pd.read_csv(CLEANED / "appliances_energy_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Group by day-period bucket as the "group" axis (mimics target grouping).
    df["period"] = df["date"].dt.hour.map(
        lambda h: 0 if pd.isna(h) else (0 if 6 <= h < 18 else 1))
    labels = {0: "Ban ngày", 1: "Ban đêm"}
    for var, lbl in [("Appliances", "Tiêu thụ (Wh)"), ("T1", "Nhiệt độ T1"),
                     ("RH_1", "Độ ẩm RH_1"), ("T_out", "Nhiệt độ ngoài"),
                     ("Windspeed", "Tốc độ gió")]:
        numeric_hist_box(df, var, "period", "appliances", lbl, labels)


def group_b_har() -> None:
    X = pd.read_csv(HAR_BASE / "train" / "X_train.txt", sep=r"\s+", header=None)
    y = pd.read_csv(HAR_BASE / "train" / "y_train.txt", sep=r"\s+", header=None)[0]
    feats = pd.read_csv(HAR_BASE / "features.txt", sep=r"\s+", header=None)[1].tolist()
    X.columns = feats[: X.shape[1]]
    X["activity"] = y.values
    # static vs dynamic grouping (1-3 dynamic, 4-6 static)
    X["grp"] = X["activity"].map(lambda a: 1 if a <= 3 else 0)
    labels = {0: "Tĩnh (4-6)", 1: "Động (1-3)"}
    sel = [("tBodyAcc-mean()-X", "Gia tốc TB-X"),
           ("tBodyAcc-std()-X", "Gia tốc std-X"),
           ("tBodyGyro-mean()-X", "Gyro TB-X"),
           ("tBodyAcc-energy()-X", "Năng lượng Acc-X"),
           ("tGravityAcc-mean()-X", "Gia tốc trọng lực-X")]
    for var, lbl in sel:
        if var in X.columns:
            numeric_hist_box(X, var, "grp", "har", lbl, labels)


def group_b_mhealth() -> None:
    frames = []
    for f in sorted(MHEALTH_DIR.glob("mHealth_subject*.log"))[:3]:
        d = pd.read_csv(f, sep=r"\s+", header=None)
        d.columns = [f"sensor_{i:02d}" for i in range(d.shape[1] - 1)] + ["label"]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"] != 0].copy()  # drop null activity
    df["grp"] = (df["label"] > 5).astype(int)  # 0 = activities 1-5, 1 = 6-12
    labels = {0: "Hoạt động 1-5", 1: "Hoạt động 6-12"}
    for var, lbl in [("sensor_01", "Acc ngực X"), ("sensor_02", "Acc ngực Y"),
                     ("sensor_03", "Acc ngực Z"), ("sensor_04", "ECG 1"),
                     ("sensor_05", "ECG 2")]:
        if var in df.columns:
            numeric_hist_box(df, var, "grp", "mhealth", lbl, labels)


# --------------------------------------------------------------------------- #
# Group C: multimedia
# --------------------------------------------------------------------------- #
def group_c_ham() -> None:
    df = pd.read_csv(CLEANED / "ham10000_metadata_cleaned.csv")
    # collapse dx into nv vs non-nv for the histogram split
    df["grp"] = (df["dx"] != "nv").astype(int)
    labels = {0: "nv (lành)", 1: "Khác nv"}
    if "age" in df.columns:
        numeric_hist_box(df, "age", "grp", "ham", "Tuổi", labels)
    # intensity features are in outputs/tables if available; else use metadata only
    intensity = ROOT / "outputs" / "tables" / "ham10000_image_intensity.csv"
    if intensity.exists():
        di = pd.read_csv(intensity)
        merge_col = "image_id" if "image_id" in di.columns else di.columns[0]
        m = df.merge(di, on=merge_col, how="inner")
        for var, lbl in [("mean_intensity", "Độ sáng TB"), ("std_intensity", "Độ phân tán")]:
            if var in m.columns:
                numeric_hist_box(m, var, "grp", "ham", lbl, labels)
    # categorical: localization and sex
    for var, lbl in [("sex", "Giới tính"), ("localization", "Vị trí tổn thương")]:
        if var in df.columns:
            categorical_bar(df, var, "grp", "ham", lbl)


def group_c_reuters() -> None:
    df = pd.read_csv(ROOT / "outputs" / "tables" / "reuters_text_features.csv")
    # top topics only for a cleaner bar; collapse to top vs other
    top = df["primary_topic"].value_counts().head(6).index.tolist()
    df["grp"] = df["primary_topic"].isin(top).astype(int)
    labels = {0: "Topic khác", 1: "Top topic"}
    for var, lbl in [("word_count", "Số từ"), ("char_count", "Số ký tự")]:
        numeric_hist_box(df, var, "grp", "reuters", lbl, labels)
    # categorical bar of top topics
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vc = df["primary_topic"].value_counts().head(10)
    ax.bar(range(len(vc)), vc.values, color=BLUE, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Số tài liệu")
    ax.set_title("Top 10 topic chính", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch2_pervar_reuters_topic_bar.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        ("Heart Disease", group_a_heart),
        ("CDC Diabetes", group_a_cdc),
        ("Breast Cancer", group_a_breast),
        ("Appliances Energy", group_b_appliances),
        ("HAR", group_b_har),
        ("MHEALTH", group_b_mhealth),
        ("HAM10000", group_c_ham),
        ("Reuters-21578", group_c_reuters),
    ]
    for name, fn in steps:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as exc:  # keep going; record which dataset failed
            print(f"[FAIL] {name}: {exc!r}")


if __name__ == "__main__":
    main()
