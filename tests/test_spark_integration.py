import pytest

pyspark = pytest.importorskip("pyspark")

from zeek_repro.data import (
    ClassificationTask,
    deterministic_class_cap,
    deterministic_paper_sample,
    stratified_split,
)
from zeek_repro.lda import sklearn_lda, sufficient_statistics_lda
from zeek_repro.metrics import evaluate_spark_predictions
from zeek_repro.preprocessing import apply_pca, fit_preprocessor
from zeek_repro.runner import _classifier, _model_diagnostics


@pytest.fixture(scope="module")
def spark():
    import os
    import sys

    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    session = (
        SparkSession.builder.master("local[1]")
        .appName("zeek-repro-tests")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.mark.integration
def test_cap_and_split_are_deterministic(spark):
    rows = [(f"u{i}", "none" if i < 10 else "Discovery", float(i), 0.0) for i in range(20)]
    frame = spark.createDataFrame(rows, ["uid", "label_tactic", "duration", "label"])
    capped = deterministic_class_cap(frame, cap=4, seed=42)
    counts = capped.groupBy("label_tactic").count().orderBy("label_tactic").collect()
    assert [row["count"] for row in counts] == [4, 4]
    first_train, _ = stratified_split(capped, 42)
    second_train, _ = stratified_split(capped, 42)
    assert sorted(row.uid for row in first_train.collect()) == sorted(row.uid for row in second_train.collect())


@pytest.mark.integration
def test_corrected_preprocessor_records_train_only_fit(spark):
    train = spark.createDataFrame(
        [("u1", "none", "tcp", 1.0, 0.0), ("u2", "Discovery", "udp", None, 1.0)],
        ["uid", "label_tactic", "proto", "duration", "label"],
    )
    test = spark.createDataFrame(
        [("u3", "Discovery", "icmp", 3.0, 1.0)],
        ["uid", "label_tactic", "proto", "duration", "label"],
    )
    result = fit_preprocessor(train, test, fit_on_all=False)
    assert result.metadata["fit_scope"] == "train-only"
    assert result.test.count() == 1
    assert "icmp" not in result.metadata["string_index_mappings"]["proto"]


@pytest.mark.integration
def test_corrected_preprocessor_drops_identifiers_and_can_skip_scaling(spark):
    rows = [
        ("u1", "none", "tcp", "a", "b", "c", 1.0, 0.0),
        ("u2", "Discovery", "udp", "d", "e", "f", 2.0, 1.0),
    ]
    frame = spark.createDataFrame(
        rows,
        ["uid", "label_tactic", "proto", "src_ip_zeek", "dest_ip_zeek", "community_id", "duration", "label"],
    )
    result = fit_preprocessor(frame, frame, feature_policy="corrected", scale=False)
    assert result.metadata["scaling"] == "none"
    assert not ({"src_ip_zeek", "dest_ip_zeek", "community_id"} & set(result.metadata["categorical_features"]))
    assert result.metadata["feature_dimension"] > 0


@pytest.mark.integration
def test_hashed_ip_policy_retains_ips_in_fixed_buckets(spark):
    rows = [
        ("u1", "none", "tcp", "10.0.0.1", "10.0.0.2", "c1", 1.0, 0.0),
        ("u2", "Discovery", "udp", "10.0.0.3", "10.0.0.4", "c2", 2.0, 1.0),
    ]
    frame = spark.createDataFrame(
        rows,
        ["uid", "label_tactic", "proto", "src_ip_zeek", "dest_ip_zeek", "community_id", "duration", "label"],
    )
    result = fit_preprocessor(frame, frame, feature_policy="hashed-ips", scale=False)
    hashing = result.metadata["hashing"]
    assert hashing["buckets"] == 256
    assert set(hashing["distinct_values"]) == {"src_ip_zeek", "dest_ip_zeek"}
    assert hashing["load_factor"] == 4 / 256
    assert "community_id" not in result.metadata["categorical_features"]


@pytest.mark.integration
def test_paper_sampling_is_exact_and_retains_discovery(spark):
    rows = []
    for label, count in (("none", 600), ("Reconnaissance", 400), ("Discovery", 20)):
        rows.extend((f"{label}-{i}", label, 0.0) for i in range(count))
    frame = spark.createDataFrame(rows, ["uid", "label_tactic", "label"])
    task = ClassificationTask(
        "Multinomial", "multinomial", ("Reconnaissance", "Discovery"),
        {"none": 0.0, "Reconnaissance": 1.0, "Discovery": 2.0},
    )
    sampled, metadata = deterministic_paper_sample(frame, "pca", task, 42)
    counts = {row.label_tactic: row["count"] for row in sampled.groupBy("label_tactic").count().collect()}
    assert sampled.count() == round(1020 * 0.05)
    assert counts["Discovery"] == 20
    assert metadata["discovery_retained"] is True


@pytest.mark.integration
def test_pca_and_lda_dimensions_and_variance(spark):
    from pyspark.ml.linalg import Vectors

    rows = []
    centers = ((0.0, 0.0, 0.0), (4.0, 0.0, 2.0), (0.0, 4.0, 6.0))
    offsets = ((0.0, 0.0, 0.0), (0.2, 0.1, -0.1), (-0.1, 0.3, 0.2), (0.1, -0.2, 0.3))
    for label, center in enumerate(centers):
        for index, offset in enumerate(offsets):
            rows.append(
                (
                    f"{label}-{index}",
                    float(label),
                    str(label),
                    Vectors.dense(*(c + delta for c, delta in zip(center, offset))),
                )
            )
    frame = spark.createDataFrame(rows, ["uid", "label", "label_tactic", "features"])
    train = frame.filter("uid not like '%-3'").drop("uid")
    test = frame.filter("uid like '%-3'").drop("uid")
    pca_train, _, pca_variance, _ = apply_pca(train, test, k=2)
    assert pca_train.first().features.size == 2
    assert 0.0 <= sum(pca_variance) <= 1.0000001
    for implementation in (sklearn_lda, sufficient_statistics_lda):
        lda_train, _, lda_variance, _ = implementation(train, test, k=2)
        assert lda_train.first().features.size == 2
        assert 0.0 <= sum(lda_variance) <= 1.0000001


@pytest.mark.integration
def test_distributed_multiclass_metrics_do_not_collect_prediction_rows(spark):
    from pyspark.ml.linalg import Vectors

    rows = [
        (0.0, 0.0, Vectors.dense(3.0, 1.0, 0.0)),
        (1.0, 1.0, Vectors.dense(0.0, 3.0, 1.0)),
        (2.0, 2.0, Vectors.dense(0.0, 1.0, 3.0)),
        (0.0, 0.0, Vectors.dense(2.0, 1.0, 0.0)),
        (1.0, 1.0, Vectors.dense(0.0, 2.0, 1.0)),
        (2.0, 2.0, Vectors.dense(0.0, 1.0, 2.0)),
    ]
    predictions = spark.createDataFrame(rows, ["label", "prediction", "rawPrediction"])
    summary, _, matrix = evaluate_spark_predictions(
        predictions, [0.0, 1.0, 2.0], distributed=True
    )
    assert summary["accuracy"] == 1.0
    assert summary["auroc"] == 1.0
    assert matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


@pytest.mark.integration
def test_linear_svc_diagnostics_are_explicit(spark):
    from pyspark.ml.linalg import Vectors

    rows = [(float(i % 2), Vectors.dense(float(i), float(i % 3))) for i in range(20)]
    frame = spark.createDataFrame(rows, ["label", "features"])
    model = _classifier("binary", max_iter=3, threshold=0.0).fit(frame)
    item = _model_diagnostics(model, max_iter=3)["submodels"][0]
    assert set(item) >= {
        "total_iterations", "objective_history", "converged", "unavailable_reason"
    }
    assert item["total_iterations"] is not None or item["unavailable_reason"]
