"""Fast Parquet inventory and study-data validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import DATASETS


def inspect_dataset(data_root: Path, dataset_name: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    spec = DATASETS[dataset_name]
    root = data_root / spec["directory"]
    files = sorted(root.rglob("*.parquet"))
    counts: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    file_items = []
    rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        file_rows = parquet.metadata.num_rows
        rows += file_rows
        file_items.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": file_rows,
            }
        )
        schemas[str(parquet.schema_arrow)] += 1
        for batch in parquet.iter_batches(columns=["label_tactic"], batch_size=262_144):
            counts.update(value.as_py() for value in batch.column(0))
    return {
        "dataset": dataset_name,
        "root": str(root),
        "files": len(files),
        "rows": rows,
        "input_files": file_items,
        "counts": dict(counts.most_common()),
        "schemas": [{"files": n, "schema": schema} for schema, n in schemas.items()],
    }


def validate_inventory(item: dict[str, Any]) -> list[str]:
    spec = DATASETS[item["dataset"]]
    problems: list[str] = []
    if item["files"] != spec["expected_files"]:
        problems.append(f"files: expected {spec['expected_files']}, found {item['files']}")
    if item["rows"] != spec["expected_rows"]:
        problems.append(f"rows: expected {spec['expected_rows']}, found {item['rows']}")
    if item["counts"] != spec["expected_counts"]:
        problems.append("label_tactic counts do not match the paper dataset")
    if len(item["schemas"]) != 1:
        problems.append(f"expected one schema, found {len(item['schemas'])}")
    return problems


def inventory(data_root: Path, datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    results = []
    for name in datasets:
        item = inspect_dataset(data_root, name)
        item["problems"] = validate_inventory(item)
        results.append(item)
    return results
