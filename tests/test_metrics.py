import math

import pytest

from zeek_repro.metrics import metrics_from_arrays


def test_binary_metrics_include_attack_and_weighted_views():
    summary, per_class, matrix = metrics_from_arrays(
        [0, 0, 1, 1], [0, 1, 1, 1], scores=[0.1, 0.7, 0.8, 0.9], labels=[0, 1]
    )
    assert summary["accuracy"] == 0.75
    assert summary["attack_precision"] == pytest.approx(2 / 3)
    assert summary["attack_recall"] == 1.0
    assert summary["auroc"] == 1.0
    assert matrix == [[1, 1], [0, 2]]
    assert sum(row["support"] for row in per_class) == 4


def test_missing_predictions_do_not_divide_by_zero():
    summary, per_class, _ = metrics_from_arrays([0, 0, 1], [0, 0, 0], labels=[0, 1])
    assert summary["attack_precision"] == 0.0
    assert summary["attack_recall"] == 0.0
    assert math.isnan(summary["auroc"])
    assert per_class[1]["f1"] == 0.0
    assert summary["attack_fpr"] == 0.0
    assert summary["collapse_flag"] is True
    assert "1.0" in summary["collapse_reason"]
    assert per_class[1]["predicted_support"] == 0
    assert summary["evaluation_valid"] is True


def test_missing_test_class_is_not_a_valid_evaluation():
    summary, _, _ = metrics_from_arrays([0, 0], [0, 0], labels=[0, 1])
    assert summary["evaluation_valid"] is False
    assert "1.0" in summary["evaluation_warning"]


def test_attack_fpr_is_positive_class_false_positive_rate():
    summary, per_class, _ = metrics_from_arrays([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    assert summary["attack_fpr"] == 0.5
    assert summary["weighted_fpr"] == pytest.approx(0.25)
    assert per_class[1]["predicted_support"] == 3


def test_multiclass_auroc_accepts_decision_scores():
    summary, _, _ = metrics_from_arrays(
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2],
        scores=[
            [3, 1, 0],
            [0, 3, 1],
            [0, 1, 3],
            [2, 1, 0],
            [0, 2, 1],
            [0, 1, 2],
        ],
        labels=[0, 1, 2],
    )
    assert summary["auroc"] == 1.0
