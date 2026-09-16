from zeek_repro.config import DATASETS
from zeek_repro.inventory import validate_inventory


def test_inventory_validation_accepts_exact_paper_counts():
    for dataset, spec in DATASETS.items():
        item = {
            "dataset": dataset,
            "files": spec["expected_files"],
            "rows": spec["expected_rows"],
            "counts": spec["expected_counts"],
            "schemas": [{"files": spec["expected_files"], "schema": "fixture"}],
        }
        assert validate_inventory(item) == []


def test_inventory_validation_reports_mismatch():
    item = {
        "dataset": "UWF-ZeekData22",
        "files": 7,
        "rows": 1,
        "counts": {},
        "schemas": [],
    }
    assert len(validate_inventory(item)) == 4

