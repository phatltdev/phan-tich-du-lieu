from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
LOG_PATH = LOGS / "ml_baselines.log"


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def classification_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in x.columns if column not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )


def classification_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced"),
    }


def load_classification_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    heart = pd.read_csv(ROOT / "outputs" / "cleaned" / "heart_disease_cleaned.csv")
    heart_y = heart["target_binary"]
    heart_x = heart.drop(columns=["num", "target_binary"])

    diabetes = pd.read_csv(ROOT / "datasets" / "uci_891_cdc_diabetes_health_indicators" / "data.csv")
    # Keep runtime predictable while preserving class balance for a baseline.
    diabetes = pd.concat(
        [
            group.sample(min(len(group), 8000), random_state=42)
            for _, group in diabetes.groupby("Diabetes_binary")
        ],
        ignore_index=True,
    )
    diabetes_y = diabetes["Diabetes_binary"]
    diabetes_x = diabetes.drop(columns=["Diabetes_binary"])

    breast = pd.read_csv(ROOT / "datasets" / "uci_17_breast_cancer_wisconsin_diagnostic" / "data.csv")
    breast_y = (breast["Diagnosis"] == "M").astype(int)
    breast_x = breast.drop(columns=["Diagnosis"])

    return {
        "heart_disease": (heart_x, heart_y),
        "cdc_diabetes_sampled": (diabetes_x, diabetes_y),
        "breast_cancer": (breast_x, breast_y),
    }


def save_confusion_plot(dataset_name: str, model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{dataset_name} - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / f"ml_{dataset_name}_{model_name}_confusion_matrix.png", dpi=180)
    plt.close()


def run_classification() -> pd.DataFrame:
    rows = []
    for dataset_name, (x, y) in load_classification_datasets().items():
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        preprocessor = classification_preprocessor(x_train)
        for model_name, model in classification_models().items():
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(x_train, y_train)
            pred = pipeline.predict(x_test)
            save_confusion_plot(dataset_name, model_name, y_test, pred)
            rows.append(
                {
                    "task": "classification",
                    "dataset": dataset_name,
                    "model": model_name,
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "accuracy": accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1_score(y_test, pred, zero_division=0),
                }
            )
            logging.info("Classification complete: %s %s", dataset_name, model_name)
    return pd.DataFrame(rows)


def run_regression() -> pd.DataFrame:
    path = ROOT / "outputs" / "cleaned" / "appliances_energy_cleaned.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    feature_cols = [
        "lights",
        "T1",
        "RH_1",
        "T2",
        "RH_2",
        "T_out",
        "Press_mm_hg",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
    ]
    x = df[feature_cols]
    y = df["Appliances"]
    split = int(len(df) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
            ]
        ),
    }

    rows = []
    prediction_frames = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        rows.append(
            {
                "task": "regression",
                "dataset": "appliances_energy",
                "model": model_name,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": rmse,
                "r2": r2_score(y_test, pred),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": df["date"].iloc[split:].values,
                    "actual": y_test.values,
                    "predicted": pred,
                    "model": model_name,
                }
            )
        )
        logging.info("Regression complete: %s", model_name)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(TABLES / "ml_appliances_energy_predictions.csv", index=False)

    plot_sample = predictions.groupby("model", group_keys=False).head(350)
    plt.figure(figsize=(12, 6))
    first_model = plot_sample["model"].iloc[0]
    first = plot_sample[plot_sample["model"] == first_model]
    plt.plot(first["date"], first["actual"], label="actual", color="black", linewidth=1)
    for model_name, group in plot_sample.groupby("model"):
        plt.plot(group["date"], group["predicted"], label=model_name, linewidth=1)
    plt.title("Appliances Energy regression prediction sample")
    plt.xlabel("Date")
    plt.ylabel("Appliances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "ml_appliances_energy_prediction_sample.png", dpi=180)
    plt.close()

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    setup_logging()
    classification = run_classification()
    regression = run_regression()
    classification.to_csv(TABLES / "ml_classification_metrics.csv", index=False)
    regression.to_csv(TABLES / "ml_regression_metrics.csv", index=False)
    pd.concat([classification, regression], ignore_index=True, sort=False).to_csv(
        TABLES / "ml_baseline_metrics.csv", index=False
    )
    print("ML baselines complete.")


if __name__ == "__main__":
    main()
