from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"


def classify(path: Path) -> str:
    parts = set(path.parts)
    if "figures" in parts:
        return "figure"
    if "tables" in parts:
        return "table"
    if "logs" in parts:
        return "log"
    if "cleaned" in parts:
        return "cleaned_data"
    return "other"


def infer_dataset(name: str) -> str:
    prefixes = [
        "heart_disease",
        "cdc_diabetes",
        "breast_cancer",
        "appliances_energy",
        "ham10000",
        "ml",
        "local_dataset",
        "uci_dataset",
        "tabular_health",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return prefix
    return "general"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in OUTPUTS.rglob("*") if path.is_file())
    rows = [
        {
            "artifact": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "artifact_type": classify(path),
            "dataset_or_scope": infer_dataset(path.name),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    pd.DataFrame(rows).to_csv(TABLES / "evidence_index.csv", index=False)
    print(f"Wrote {TABLES.relative_to(ROOT) / 'evidence_index.csv'} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
