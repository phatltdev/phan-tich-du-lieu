from __future__ import annotations

import json
import shutil
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUTPUTS = ROOT / "outputs"
LOGS = OUTPUTS / "logs"
TABLES = OUTPUTS / "tables"
LOG_PATH = LOGS / "ham10000_prepare.log"


def write_log(message: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    target_dir = DATASETS / "ham10000"
    target_dir.mkdir(exist_ok=True)

    try:
        write_log("Downloading HAM10000 from Kaggle via kagglehub.")
        source_path = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
        write_log(f"kagglehub source path: {source_path}")

        marker = target_dir / "source_path.txt"
        marker.write_text(str(source_path), encoding="utf-8")

        metadata_files = list(source_path.glob("*metadata*.csv")) + list(source_path.glob("*.csv"))
        copied_metadata = []
        for metadata in metadata_files:
            destination = target_dir / metadata.name
            shutil.copy2(metadata, destination)
            copied_metadata.append(destination.name)

        summary = {
            "dataset": "Skin Cancer MNIST HAM10000",
            "group": "C",
            "status": "available_via_kagglehub_cache",
            "source_path": str(source_path),
            "local_path": str(target_dir.relative_to(ROOT)),
            "metadata_files_copied": copied_metadata,
            "note": "Images may remain in the Kaggle cache path to avoid duplicating large files.",
        }
        (target_dir / "ham10000_inventory.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_log("HAM10000 preparation completed.")
        print("HAM10000 preparation completed.")
    except Exception as exc:
        failure = {
            "dataset": "Skin Cancer MNIST HAM10000",
            "group": "C",
            "status": "blocked",
            "error": repr(exc),
            "next_step": "Configure Kaggle credentials or manually download the dataset.",
        }
        (target_dir / "ham10000_inventory.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_log(f"HAM10000 preparation failed: {exc!r}")
        print("HAM10000 preparation failed. See outputs/logs/ham10000_prepare.log")


if __name__ == "__main__":
    main()
