import json

import pandas as pd

from zeek_repro.comparison import compare_run, grade_accuracy
from zeek_repro.paper_reference import reference_rows


def test_reference_fixture_covers_tables_5_through_22_and_marks_errata():
    rows = reference_rows()
    assert {row["table"] for row in rows} == set(range(5, 23))
    assert all(row["reference_version"] == 1 for row in rows)
    assert any(row["table"] == 10 and row["status"] == "excluded" for row in rows)
    assert any(
        row["table"] == 6 and row["metric"] == "weighted_f1" and row["status"] == "excluded"
        for row in rows
    )


def test_accuracy_grades_have_inclusive_boundaries():
    assert grade_accuracy(1.0) == "Excellent"
    assert grade_accuracy(3.0) == "Close"
    assert grade_accuracy(5.0) == "Directional"
    assert grade_accuracy(5.01) == "Divergent"


def test_compare_writes_machine_readable_and_narrative_outputs(tmp_path):
    pd.DataFrame([
        {
            "dataset": "UWF-ZeekData22", "task": "Reconnaissance", "preprocessing": "minimal",
            "k": "", "accuracy": .9993, "weighted_precision": .9993,
            "weighted_recall": .9993, "weighted_f1": .9992, "attack_fpr": .0014,
            "auroc": .9992, "collapse_flag": False, "collapse_reason": "",
        }
    ]).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "config": {"method": "paper-compatible", "profile": "development"},
        "protocol": {"protocol_version": 2, "feature_policy": "legacy-ordinal"},
    }))
    outputs = compare_run(tmp_path)
    assert all(path.exists() for path in outputs)
    comparison = pd.read_csv(tmp_path / "paper_comparison.csv")
    matched = comparison[
        (comparison.table == 5) & (comparison.task == "Reconnaissance") & (comparison.metric == "accuracy")
    ].iloc[0]
    assert matched.comparison_status == "comparable"
    assert matched.grade == "Excellent"
    audit = (tmp_path / "REPRODUCTION_AUDIT.md").read_text()
    assert "Median absolute error" in audit
    assert "Accuracy by method and task" in audit
    assert "Variance comparison" in audit
    assert "Local timing ranks" in audit


def test_paper_compatible_minimal_baseline_stays_within_point_one_pp(tmp_path):
    """Lock the empirically reproduced Table 5 baseline used for acceptance."""
    pd.DataFrame([
        {
            "dataset": "UWF-ZeekData22", "task": task, "preprocessing": "minimal",
            "k": "", "accuracy": accuracy, "collapse_flag": False,
            "collapse_reason": "", "evaluation_valid": True,
        }
        for task, accuracy in (
            ("Reconnaissance", 1.00000),
            ("Discovery", 0.99987),
            ("Multinomial", 0.99977),
        )
    ]).to_csv(tmp_path / "metrics.csv", index=False)

    compare_run(tmp_path)
    comparison = pd.read_csv(tmp_path / "paper_comparison.csv")
    baseline = comparison[
        (comparison.table == 5)
        & (comparison.metric == "accuracy")
        & (comparison.comparison_status == "comparable")
    ]
    assert len(baseline) == 3
    assert baseline.absolute_percentage_points.mean() <= 0.10
