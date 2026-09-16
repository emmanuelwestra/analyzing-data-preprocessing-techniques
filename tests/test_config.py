from pathlib import Path

from zeek_repro.config import DATASETS, PCA_COMPONENTS, RunConfig
from zeek_repro.data import tasks_for


def test_paper_component_grid_is_preserved():
    assert PCA_COMPONENTS == (2, 3, 5, 10, 11)


def test_dataset_tasks_have_benign_zero_and_attack_one():
    for dataset in DATASETS:
        tasks = tasks_for(dataset)
        for task in tasks[:-1]:
            assert task.label_mapping["none"] == 0.0
            assert task.label_mapping[task.tactics[0]] == 1.0
        assert len(tasks[-1].label_mapping) == len(tasks[-1].tactics) + 1


def test_protocol_defaults_separate_fidelity_and_corrected_science():
    common = dict(
        data_root=Path("data/raw"), output_root=Path("results"), stage="all",
        profile="development", datasets=("UWF-ZeekData22",),
    )
    paper = RunConfig(method="paper-compatible", **common).protocol()
    corrected = RunConfig(method="corrected", **common).protocol()
    assert (paper["svm_max_iter"], paper["svm_threshold"]) == (10, 0.00001)
    assert paper["preprocessing_fit_scope"] == "train+test"
    assert corrected["feature_policy"] == "corrected"
    assert (corrected["svm_max_iter"], corrected["svm_threshold"]) == (100, 0.0)
