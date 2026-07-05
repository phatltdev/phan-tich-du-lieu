from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
LOGS_DIR = ROOT / "outputs" / "logs"
TABLES_DIR = ROOT / "outputs" / "tables"
LOG_PATH = LOGS_DIR / "uci_dataset_download.log"

DATASETS = [
    {"id": 891, "group": "A", "short_name": "cdc_diabetes_health_indicators"},
    {"id": 17, "group": "A", "short_name": "breast_cancer_wisconsin_diagnostic"},
    {"id": 374, "group": "B", "short_name": "appliances_energy_prediction"},
    {"id": 319, "group": "B", "short_name": "mhealth"},
    {"id": 137, "group": "C", "short_name": "reuters_21578"},
]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def as_frame(value: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.Series):
        return value.to_frame()
    return value.copy()


def save_dataset(entry: dict[str, object]) -> dict[str, object]:
    uci_id = int(entry["id"])
    fallback_name = str(entry["short_name"])
    group = str(entry["group"])

    logging.info("Fetching UCI dataset id=%s", uci_id)
    dataset = fetch_ucirepo(id=uci_id)
    official_name = dataset.metadata.name or fallback_name
    folder = DATASETS_DIR / f"uci_{uci_id}_{slugify(official_name)}"
    folder.mkdir(parents=True, exist_ok=True)

    features = as_frame(dataset.data.features)
    targets = as_frame(dataset.data.targets)

    if not features.empty and not targets.empty:
        full = pd.concat([features, targets], axis=1)
    elif not features.empty:
        full = features
    else:
        full = targets

    features.to_csv(folder / "features.csv", index=False)
    targets.to_csv(folder / "targets.csv", index=False)
    full.to_csv(folder / "data.csv", index=False)

    metadata = dict(dataset.metadata)
    variables = dataset.variables
    with (folder / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)
    variables.to_csv(folder / "variables.csv", index=False)

    row = {
        "uci_id": uci_id,
        "group": group,
        "short_name": fallback_name,
        "official_name": official_name,
        "local_path": folder.relative_to(ROOT).as_posix(),
        "samples": int(full.shape[0]),
        "feature_columns": int(features.shape[1]),
        "target_columns": int(targets.shape[1]),
        "total_columns": int(full.shape[1]),
        "has_missing_values": metadata.get("has_missing_values"),
        "target_col": metadata.get("target_col"),
        "repository_url": metadata.get("repository_url"),
        "data_url": metadata.get("data_url"),
    }
    logging.info("Saved %s rows x %s columns to %s", full.shape[0], full.shape[1], folder)
    return row


def main() -> None:
    ensure_dirs()
    setup_logging()

    rows = []
    failures = []
    for entry in DATASETS:
        try:
            rows.append(save_dataset(entry))
        except Exception as exc:  # keep downloading other datasets when one source fails
            failure = {
                "uci_id": entry["id"],
                "group": entry["group"],
                "short_name": entry["short_name"],
                "error": repr(exc),
            }
            failures.append(failure)
            logging.exception("Failed to download %s", entry)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "uci_dataset_inventory.csv", index=False)
    pd.DataFrame(failures).to_csv(TABLES_DIR / "uci_dataset_download_failures.csv", index=False)

    print(f"Downloaded {len(rows)} UCI datasets; {len(failures)} failures. See {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
