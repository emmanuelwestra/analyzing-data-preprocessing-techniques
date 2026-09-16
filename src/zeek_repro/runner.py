"""Experiment orchestration and artifact persistence."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import DATASETS, PCA_COMPONENTS, RunConfig
from .data import (
    build_task_frame,
    class_counts,
    create_spark,
    load_dataset,
    paper_split,
    require_java_17,
    sample_for_run,
    stratified_split,
    tasks_for,
)
from .inventory import inventory
from .lda import sklearn_lda, sufficient_statistics_lda
from .metrics import evaluate_spark_predictions
from .preprocessing import apply_pca, fit_preprocessor


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _classifier(task_kind: str, max_iter: int, threshold: float):
    from pyspark.ml.classification import LinearSVC, OneVsRest

    base = LinearSVC(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=0.0,
        fitIntercept=True,
        threshold=threshold,
    )
    return base if task_kind == "binary" else OneVsRest(classifier=base, parallelism=1)


def _model_diagnostics(model, max_iter: int) -> dict:
    models = list(getattr(model, "models", [model]))
    items = []
    for index, fitted in enumerate(models):
        unavailable_reason = ""
        try:
            summary = fitted._java_obj.summary()
            history = [float(value) for value in summary.objectiveHistory()]
            total = int(summary.totalIterations())
        except Exception as exc:
            summary = getattr(fitted, "summary", None)
            history = list(getattr(summary, "objectiveHistory", []) or [])
            total = getattr(summary, "totalIterations", None)
            if total is None and not history:
                unavailable_reason = f"training summary unavailable: {type(exc).__name__}"
        items.append(
            {
                "submodel": index,
                "total_iterations": int(total) if total is not None else None,
                "objective_history": [float(value) for value in history],
                "converged": bool(total < max_iter) if total is not None else None,
                "unavailable_reason": unavailable_reason,
            }
        )
    return {"max_iter": max_iter, "submodels": items}


def _fit_and_measure(
    train,
    test,
    task_kind,
    labels,
    repetitions,
    warmup,
    distributed_metrics,
    max_iter,
    threshold,
):
    def once(evaluate: bool):
        training_started = perf_counter()
        model = _classifier(task_kind, max_iter, threshold).fit(train)
        training_seconds = perf_counter() - training_started
        prediction_started = perf_counter()
        predictions = model.transform(test).cache()
        predictions.count()
        prediction_seconds = perf_counter() - prediction_started
        metric_started = perf_counter()
        evaluated = (
            evaluate_spark_predictions(predictions, labels, distributed=distributed_metrics)
            if evaluate
            else None
        )
        metric_seconds = perf_counter() - metric_started
        predictions.unpersist()
        return (
            training_seconds,
            prediction_seconds,
            metric_seconds,
            evaluated,
            _model_diagnostics(model, max_iter),
        )

    if warmup:
        once(evaluate=False)
    timings = []
    evaluated = None
    for repetition in range(1, repetitions + 1):
        training, prediction, metric, evaluated, diagnostics = once(
            evaluate=repetition == repetitions
        )
        timings.append(
            {
                "repetition": repetition,
                "training_seconds": training,
                "prediction_seconds": prediction,
                "metric_seconds": metric,
                "evaluation_seconds": prediction + metric,
            }
        )
    return timings, evaluated, diagnostics


def _variant_frames(stage, prepared, task, profile, method):
    fit_on_all = method == "paper-compatible"
    if stage == "svm":
        yield "minimal", None, prepared.train, prepared.test, [], 0.0, "spark"
    elif stage == "pca":
        dimension = prepared.train.select("features").first().features.size
        for k in PCA_COMPONENTS:
            if k > dimension:
                continue
            train, test, variance, seconds = apply_pca(
                prepared.train, prepared.test, k, fit_on_all=fit_on_all
            )
            yield "pca", k, train, test, variance, seconds, "spark"
    elif stage == "lda":
        class_count = len(task.label_mapping)
        for k in range(1, class_count):
            if task.kind == "binary" and k > 1:
                break
            if profile == "development":
                train, test, variance, seconds = sklearn_lda(
                    prepared.train, prepared.test, k, fit_on_all=fit_on_all
                )
                engine = "scikit-learn-svd"
            else:
                train, test, variance, seconds = sufficient_statistics_lda(
                    prepared.train, prepared.test, k, fit_on_all=fit_on_all
                )
                engine = "local-sufficient-statistics"
            yield "lda", k, train, test, variance, seconds, engine


def run_experiments(config: RunConfig) -> Path:
    # Check before creating an incomplete results directory.
    require_java_17()
    selected_stages = ("svm", "pca", "lda") if config.stage == "all" else (config.stage,)
    inventory_items = inventory(config.data_root, config.datasets)
    if config.profile == "full":
        problems = [f"{item['dataset']}: {problem}" for item in inventory_items for problem in item["problems"]]
        if problems:
            raise RuntimeError("Full run refused because inventory validation failed: " + "; ".join(problems))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{config.stage}-{config.profile}-{config.method}"
    final_run_dir = config.output_root / run_id
    run_dir = config.output_root / f"{run_id}.incomplete"
    (run_dir / "confusion_matrices").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "inventory.json").write_text(
        json.dumps(inventory_items, indent=2, default=_json_default), encoding="utf-8"
    )

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")
    metrics_rows: list[dict] = []
    timing_rows: list[dict] = []
    variance_rows: list[dict] = []
    diagnostic_sweep_rows: list[dict] = []
    model_diagnostics: list[dict] = []
    preprocessing_metadata: list[dict] = []
    protocol = config.protocol()
    try:
        for dataset_name in config.datasets:
            source = load_dataset(spark, config.data_root, dataset_name)
            for task in tasks_for(dataset_name):
                task_frame = build_task_frame(source, task)
                development_frame = None
                development_sampling = None
                if config.profile == "development":
                    development_frame, development_sampling = sample_for_run(
                        task_frame,
                        dataset_name,
                        "minimal",
                        task,
                        config.profile,
                        config.method,
                        config.development_cap,
                        config.seed,
                    )
                    development_frame = development_frame.cache()
                    development_frame.count()
                for stage in selected_stages:
                    preprocessing_name = "minimal" if stage == "svm" else stage
                    if development_frame is not None:
                        frame, sampling = development_frame, development_sampling
                    else:
                        frame, sampling = sample_for_run(
                            task_frame,
                            dataset_name,
                            preprocessing_name,
                            task,
                            config.profile,
                            config.method,
                            config.development_cap,
                            config.seed,
                        )
                    if config.method == "corrected":
                        raw_train, raw_test = stratified_split(frame, config.seed)
                    else:
                        raw_train, raw_test = paper_split(frame, dataset_name, task, config.seed)
                    scale = (
                        preprocessing_name != "minimal"
                        or protocol["minimal_scaling"] == "scaled"
                    )
                    prepared = fit_preprocessor(
                        raw_train,
                        raw_test,
                        fit_on_all=protocol["preprocessing_fit_scope"] == "train+test",
                        feature_policy=protocol["feature_policy"],
                        scale=scale,
                    )
                    preprocessing_metadata.append(
                        {
                            "dataset": dataset_name,
                            "task": task.name,
                            "stage": preprocessing_name,
                            "sampling": sampling,
                            "train_class_counts": class_counts(raw_train),
                            "test_class_counts": class_counts(raw_test),
                            **prepared.metadata,
                        }
                    )
                    for variant, k, train, test, variance, reduction_seconds, engine in _variant_frames(
                        stage, prepared, task, config.profile, config.method
                    ):
                        labels = sorted(task.label_mapping.values())
                        timings, evaluated, diagnostics = _fit_and_measure(
                            train,
                            test,
                            task.kind,
                            labels,
                            config.repetitions,
                            config.warmup,
                            distributed_metrics=config.profile == "full",
                            max_iter=protocol["svm_max_iter"],
                            threshold=protocol["svm_threshold"],
                        )
                        summary, per_class, matrix = evaluated
                        identity = {
                            "dataset": dataset_name,
                            "task": task.name,
                            "task_kind": task.kind,
                            "method": config.method,
                            "profile": config.profile,
                            "preprocessing": variant,
                            "k": k if k is not None else "",
                            "engine": engine,
                            "execution_mode": "local-spark",
                            "accelerator": "cpu",
                            "distributed": False,
                        }
                        metrics_rows.append({**identity, **summary})
                        model_diagnostics.append({**identity, **diagnostics})
                        for row in timings:
                            timing_rows.append(
                                {
                                    **identity,
                                    "preprocessing_seconds": prepared.seconds + reduction_seconds,
                                    "base_preprocessing_seconds": prepared.seconds,
                                    "dimensionality_reduction_seconds": reduction_seconds,
                                    **row,
                                }
                            )
                        cumulative = 0.0
                        for component, value in enumerate(variance, start=1):
                            cumulative += value
                            variance_rows.append(
                                {
                                    **identity,
                                    "component": component,
                                    "explained_variance": value,
                                    "cumulative_variance": cumulative,
                                }
                            )
                        matrix_path = run_dir / "confusion_matrices" / (
                            f"{dataset_name}-{task.name}-{variant}-k{k or 'na'}.json"
                        )
                        matrix_path.write_text(
                            json.dumps(
                                {
                                    "identity": identity,
                                    "label_mapping": task.label_mapping,
                                    "matrix": matrix,
                                    "per_class": per_class,
                                },
                                indent=2,
                                default=_json_default,
                            ),
                            encoding="utf-8",
                        )
                        if config.diagnostic_sweep and summary["collapse_flag"]:
                            for diagnostic_iter in (10, 50, 100, 200):
                                for diagnostic_threshold in (0.00001, 0.0):
                                    sweep_timings, sweep_evaluated, sweep_diagnostics = _fit_and_measure(
                                        train,
                                        test,
                                        task.kind,
                                        labels,
                                        repetitions=1,
                                        warmup=False,
                                        distributed_metrics=config.profile == "full",
                                        max_iter=diagnostic_iter,
                                        threshold=diagnostic_threshold,
                                    )
                                    sweep_summary, _, _ = sweep_evaluated
                                    diagnostic_sweep_rows.append(
                                        {
                                            **identity,
                                            "svm_max_iter": diagnostic_iter,
                                            "svm_threshold": diagnostic_threshold,
                                            **sweep_summary,
                                            **sweep_timings[0],
                                        }
                                    )
                                    model_diagnostics.append(
                                        {
                                            **identity,
                                            "diagnostic_sweep": True,
                                            "svm_threshold": diagnostic_threshold,
                                            **sweep_diagnostics,
                                        }
                                    )
                    prepared.train.unpersist()
                    prepared.test.unpersist()
                if development_frame is not None:
                    development_frame.unpersist()
    finally:
        spark.stop()

    _write_csv(run_dir / "metrics.csv", metrics_rows)
    _write_csv(run_dir / "timings.csv", timing_rows)
    _write_csv(run_dir / "variance.csv", variance_rows)
    _write_csv(run_dir / "diagnostic_sweeps.csv", diagnostic_sweep_rows)
    with (run_dir / "model_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for item in model_diagnostics:
            stream.write(json.dumps(item, default=_json_default) + "\n")
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "manifest_version": 2,
        "protocol": protocol,
        "execution_mode": "local-spark",
        "accelerator": "cpu",
        "distributed": False,
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "software": {
            "python": sys.version,
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "pyspark",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "matplotlib",
                    "seaborn",
                )
            },
        },
        "git_revision": git_revision,
        "preprocessing": preprocessing_metadata,
        "limitations": [
            "CPU-only local run; no CUDA, RAPIDS, or distributed MapReduce measurements.",
            "Paper-compatible settings are an interpretation because original code was not published.",
            "Absolute timings are not compared with the paper unless hardware, sample size, and timing boundaries match.",
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    run_dir.rename(final_run_dir)
    return final_run_dir
