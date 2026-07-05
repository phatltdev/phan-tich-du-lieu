from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def extended_summary(df: pd.DataFrame, columns: list[str], name_col: str = "variable") -> pd.DataFrame:
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mean = s.mean()
        std = s.std(ddof=1)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        rows.append({
            name_col: col,
            "count": int(s.count()),
            "mean": mean,
            "median": s.median(),
            "min": s.min(),
            "max": s.max(),
            "range": s.max() - s.min(),
            "variance": s.var(ddof=1),
            "std": std,
            "cv": std / mean if mean != 0 else np.nan,
            "p25": q1,
            "p75": q3,
            "iqr": q3 - q1,
        })
    return pd.DataFrame(rows)


def add_dispersion(path: Path, out_path: Path, name_col: str) -> None:
    df = pd.read_csv(path)
    rename = {"25%": "p25", "50%": "median", "75%": "p75"}
    df = df.rename(columns=rename)
    df["range"] = df["max"] - df["min"]
    df["variance"] = df["std"] ** 2
    df["cv"] = df["std"] / df["mean"].replace(0, np.nan)
    df["iqr"] = df["p75"] - df["p25"]
    first = [name_col, "count", "mean", "median", "min", "max", "range", "variance", "std", "cv", "p25", "p75", "iqr"]
    df[first].to_csv(out_path, index=False)


def write_variable_types() -> None:
    tables = {
        "appliances_energy_variable_types.csv": [
            ("date", "datetime", "time_index", "Mốc thời gian 10 phút"),
            ("Appliances", "continuous", "target", "Năng lượng thiết bị"),
            ("lights", "continuous", "feature", "Năng lượng chiếu sáng"),
            ("T1..T9", "continuous", "feature", "Nhiệt độ các khu vực"),
            ("RH_1..RH_9", "continuous", "feature", "Độ ẩm các khu vực"),
            ("T_out,RH_out,Windspeed", "continuous", "feature", "Điều kiện thời tiết"),
        ],
        "har_variable_types.csv": [
            ("activity", "categorical", "target", "6 nhãn hoạt động"),
            ("tBodyAcc-*", "continuous", "feature", "Đặc trưng gia tốc miền thời gian"),
            ("tBodyGyro-*", "continuous", "feature", "Đặc trưng con quay miền thời gian"),
            ("fBodyAcc-*", "continuous", "feature", "Đặc trưng gia tốc miền tần số"),
            ("subject", "categorical", "group", "Mã người tham gia"),
        ],
        "mhealth_variable_types.csv": [
            ("activity", "categorical", "target", "Nhãn hoạt động/NULL"),
            ("sensor_01..sensor_23", "continuous", "feature", "Kênh cảm biến gia tốc/gyro/từ trường/sinh lý"),
            ("subject", "categorical", "group", "Mã người tham gia"),
        ],
        "ham10000_variable_types.csv": [
            ("dx", "categorical", "target", "7 lớp chẩn đoán"),
            ("age", "continuous", "metadata", "Tuổi bệnh nhân"),
            ("sex", "categorical", "metadata", "Giới tính"),
            ("localization", "categorical", "metadata", "Vị trí tổn thương"),
            ("mean_intensity,std_intensity", "continuous", "image_feature", "Đặc trưng ảnh số hóa"),
        ],
        "reuters_variable_types.csv": [
            ("primary_topic", "categorical", "target", "Chủ đề chính"),
            ("word_count", "discrete", "feature", "Số từ"),
            ("char_count", "discrete", "feature", "Số ký tự"),
            ("topics", "multi-label", "metadata", "Danh sách topic"),
        ],
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows, columns=["variable", "type", "role", "description"]).to_csv(TABLES / filename, index=False)


def main() -> None:
    add_dispersion(TABLES / "har_selected_feature_summary.csv", TABLES / "har_feature_summary_extended.csv", "feature")
    add_dispersion(TABLES / "mhealth_selected_sensor_summary.csv", TABLES / "mhealth_sensor_summary_extended.csv", "sensor")

    ham = pd.read_csv(TABLES / "ham10000_image_feature_sample.csv")
    meta = pd.read_csv(ROOT / "outputs" / "cleaned" / "ham10000_metadata_cleaned.csv")
    ham = ham.assign(age=pd.to_numeric(meta["age"], errors="coerce").sample(min(len(ham), len(meta)), random_state=42).to_numpy()) if "age" not in ham else ham
    extended_summary(ham, ["width", "height", "mean_intensity", "std_intensity", "age"]).to_csv(TABLES / "ham10000_numeric_summary.csv", index=False)

    reuters = pd.read_csv(TABLES / "reuters_text_features.csv")
    extended_summary(reuters, ["word_count", "char_count"]).to_csv(TABLES / "reuters_text_summary.csv", index=False)
    write_variable_types()


if __name__ == "__main__":
    main()
