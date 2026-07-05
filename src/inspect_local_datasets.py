from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
TABLES = ROOT / "outputs" / "tables"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        return sum(1 for _ in file)


def inspect_har() -> dict[str, object]:
    base = DATASETS / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
    features = count_lines(base / "features.txt")
    train_rows = count_lines(base / "train" / "X_train.txt")
    test_rows = count_lines(base / "test" / "X_test.txt")
    return {
        "dataset": "Human Activity Recognition Using Smartphones",
        "group": "B",
        "local_path": base.relative_to(ROOT).as_posix(),
        "samples": train_rows + test_rows,
        "features": features,
        "targets": 1,
        "status": "available_extracted",
        "notes": "Feature vectors plus raw inertial signal files are available.",
    }


def inspect_mhealth() -> dict[str, object]:
    base = DATASETS / "MHEALTHDATASET" / "MHEALTHDATASET"
    files = sorted(base.glob("mHealth_subject*.log"))
    first = pd.read_csv(files[0], sep=r"\s+", header=None)
    total_rows = sum(count_lines(path) for path in files)
    return {
        "dataset": "MHEALTH Dataset",
        "group": "B",
        "local_path": base.relative_to(ROOT).as_posix(),
        "samples": total_rows,
        "features": int(first.shape[1] - 1),
        "targets": 1,
        "status": "available_extracted",
        "notes": f"{len(files)} subject log files; last column is activity label.",
    }


def inspect_reuters() -> dict[str, object]:
    base = DATASETS / "reuters21578"
    sgm_files = glob.glob(str(base / "*.sgm"))
    return {
        "dataset": "Reuters-21578 Text Categorization Collection",
        "group": "C",
        "local_path": base.relative_to(ROOT).as_posix(),
        "samples": 21578,
        "features": 5,
        "targets": "multi-label categories",
        "status": "available_extracted",
        "notes": f"{len(sgm_files)} SGML files plus category metadata files.",
    }


def inspect_ham10000() -> dict[str, object]:
    base = DATASETS / "ham10000"
    metadata = pd.read_csv(base / "HAM10000_metadata.csv")
    return {
        "dataset": "Skin Cancer MNIST HAM10000",
        "group": "C",
        "local_path": base.relative_to(ROOT).as_posix(),
        "samples": len(metadata),
        "features": len(metadata.columns),
        "targets": "dx",
        "status": "available_via_kagglehub_cache",
        "notes": "Metadata copied locally; original JPG images are in KaggleHub cache path recorded in source_path.txt.",
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = [inspect_har(), inspect_mhealth(), inspect_reuters(), inspect_ham10000()]
    pd.DataFrame(rows).to_csv(TABLES / "local_dataset_inventory.csv", index=False)
    print(f"Wrote {TABLES.relative_to(ROOT) / 'local_dataset_inventory.csv'}")


if __name__ == "__main__":
    main()
