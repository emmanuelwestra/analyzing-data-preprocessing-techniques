"""Compare completed runs with the versioned published reference fixture."""

from __future__ import annotations

import json
from pathlib import Path

from .paper_reference import ERRATA, REFERENCE_VERSION, reference_rows


def grade_accuracy(abs_percentage_points: float) -> str:
    if abs_percentage_points <= 1:
        return "Excellent"
    if abs_percentage_points <= 3:
        return "Close"
    if abs_percentage_points <= 5:
        return "Directional"
    return "Divergent"


def _key(dataset, preprocessing, task, k, metric):
    return (dataset, preprocessing, task, -1 if k is None or str(k) == "nan" else int(float(k)), metric)


def _reproduced_values(run_dir: Path) -> tuple[dict[tuple, float], set[tuple], set[tuple]]:
    import pandas as pd

    values = {}
    collapsed = set()
    invalid = set()
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size:
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            identity = _key(row.dataset, row.preprocessing, row.task, row.k, "")[:-1]
            if str(row.get("collapse_flag", False)).lower() == "true":
                collapsed.add(identity)
            if str(row.get("evaluation_valid", True)).lower() == "false":
                invalid.add(identity)
            for metric in (
                "accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
                "attack_fpr", "auroc",
            ):
                if metric in metrics and pd.notna(row.get(metric)):
                    values[_key(row.dataset, row.preprocessing, row.task, row.k, metric)] = float(row[metric])

    variance_path = run_dir / "variance.csv"
    if variance_path.exists() and variance_path.stat().st_size:
        variance = pd.read_csv(variance_path)
        if not variance.empty:
            maximum = variance.groupby(["dataset", "preprocessing", "task"])["k"].transform("max")
            for _, row in variance[variance.k == maximum].iterrows():
                for metric in ("explained_variance", "cumulative_variance"):
                    values[_key(row.dataset, row.preprocessing, row.task, row.component, metric)] = float(row[metric])

    timing_path = run_dir / "timings.csv"
    if timing_path.exists() and timing_path.stat().st_size:
        timings = pd.read_csv(timing_path)
        mapping = {
            "preprocessing_seconds": "preprocessing_seconds",
            "cpu_training_seconds": "training_seconds",
            "cpu_testing_seconds": "evaluation_seconds",
        }
        grouped = timings.groupby(["dataset", "preprocessing", "task", "k"], dropna=False).median(numeric_only=True).reset_index()
        for _, row in grouped.iterrows():
            for reference_metric, local_metric in mapping.items():
                if local_metric in grouped and pd.notna(row.get(local_metric)):
                    values[_key(row.dataset, row.preprocessing, row.task, row.k, reference_metric)] = float(row[local_metric])
    return values, collapsed, invalid


def comparison_rows(run_dir: Path) -> list[dict]:
    reproduced, collapsed, invalid = _reproduced_values(run_dir)
    output = []
    for row in reference_rows():
        result = dict(row)
        value = reproduced.get(_key(row["dataset"], row["preprocessing"], row["task"], row["k"], row["metric"]))
        result["reproduced_value"] = value
        identity = _key(row["dataset"], row["preprocessing"], row["task"], row["k"], "")[:-1]
        result["collapse_flag"] = identity in collapsed
        result["evaluation_valid"] = identity not in invalid
        if row["status"] == "excluded":
            comparison_status = "excluded-paper-erratum"
        elif row["metric"].startswith("gpu_") or "_mr_" in row["metric"]:
            comparison_status = "unmeasured-backend"
        elif value is None:
            comparison_status = "missing-run-result"
        elif identity in invalid:
            comparison_status = "invalid-run-result"
        elif identity in collapsed and row["unit"] == "fraction":
            comparison_status = "collapsed-run-result"
        elif row["unit"] == "seconds":
            comparison_status = "context-only-hardware-mismatch"
        else:
            comparison_status = "comparable"
        result["comparison_status"] = comparison_status
        result["difference"] = value - row["paper_value"] if value is not None else None
        result["absolute_difference"] = abs(result["difference"]) if value is not None else None
        result["absolute_percentage_points"] = (
            result["absolute_difference"] * 100
            if value is not None and row["unit"] == "fraction"
            else None
        )
        result["grade"] = (
            grade_accuracy(result["absolute_percentage_points"])
            if row["metric"] == "accuracy" and comparison_status == "comparable"
            else ""
        )
        output.append(result)
    return output


def _audit_markdown(run_dir: Path, frame) -> str:
    import pandas as pd

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    accuracy = frame[(frame.metric == "accuracy") & (frame.comparison_status == "comparable")].copy()
    collapsed = pd.DataFrame()
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size:
        metrics = pd.read_csv(metrics_path)
        if "collapse_flag" in metrics:
            collapsed = metrics[metrics.collapse_flag.astype(str).str.lower().eq("true")]

    lines = [
        "# Reproduction audit",
        "",
        "## Verdict",
        "",
    ]
    if accuracy.empty:
        lines.append("No valid published accuracy rows matched this run. The run is not yet statistically comparable.")
    else:
        errors = accuracy.absolute_percentage_points
        lines.append(
            f"Matched {len(accuracy)} valid accuracy rows. Median absolute error is {errors.median():.2f} percentage points "
            f"and mean absolute error is {errors.mean():.2f} percentage points."
        )
        lines += [
            "",
            f"- Within 1 pp: {(errors <= 1).mean() * 100:.1f}%",
            f"- Within 3 pp: {(errors <= 3).mean() * 100:.1f}%",
            f"- Within 5 pp: {(errors <= 5).mean() * 100:.1f}%",
        ]
        lines += ["", "### Accuracy by method and task", ""]
        accuracy_groups = (
            accuracy.groupby(["dataset", "preprocessing", "task"])
            .absolute_percentage_points
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        lines += [
            "| Dataset | Method | Task | N | Median error (pp) | Mean error (pp) |",
            "|---|---|---|---:|---:|---:|",
        ]
        for _, row in accuracy_groups.iterrows():
            lines.append(
                f"| {row.dataset} | {row.preprocessing} | {row.task} | {int(row['count'])} "
                f"| {row['median']:.2f} | {row['mean']:.2f} |"
            )

    variance = frame[
        frame.metric.isin(("explained_variance", "cumulative_variance"))
        & frame.comparison_status.eq("comparable")
    ].copy()
    lines += ["", "## Variance comparison", ""]
    if variance.empty:
        lines.append("No valid published variance cells matched this run.")
    else:
        variance_groups = (
            variance.groupby(["dataset", "preprocessing"])
            .absolute_percentage_points
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        lines += [
            "| Dataset | Method | N | Median error (pp) | Mean error (pp) |",
            "|---|---|---:|---:|---:|",
        ]
        for _, row in variance_groups.iterrows():
            lines.append(
                f"| {row.dataset} | {row.preprocessing} | {int(row['count'])} "
                f"| {row['median']:.2f} | {row['mean']:.2f} |"
            )

    timing_path = run_dir / "timings.csv"
    lines += ["", "## Local timing ranks", ""]
    if not timing_path.exists() or not timing_path.stat().st_size:
        lines.append("No local timing observations were available.")
    else:
        timings = pd.read_csv(timing_path)
        phases = [
            phase for phase in (
                "preprocessing_seconds", "dimensionality_reduction_seconds",
                "training_seconds", "prediction_seconds", "metric_seconds",
            ) if phase in timings
        ]
        if not phases:
            lines.append("No phase-level timing observations were available.")
        else:
            medians = timings.groupby(
                ["dataset", "task", "preprocessing"], as_index=False
            )[phases].median(numeric_only=True)
            rankable = [
                (identity, group)
                for identity, group in medians.groupby(["dataset", "task"])
                if group.preprocessing.nunique() >= 2
            ]
            if not rankable:
                lines.append("Only one preprocessing method was measured per task, so no within-run rank is available.")
            else:
                lines += [
                    "Ranks are based only on this run's median seconds; absolute deltas against the paper are not scored.",
                    "",
                    "| Dataset | Task | Phase | Fastest to slowest |",
                    "|---|---|---|---|",
                ]
                for (dataset, task), group in rankable:
                    for phase in phases:
                        ordered = group.sort_values(phase)
                        ranking = " → ".join(
                            f"{row.preprocessing} ({row[phase]:.3f}s)"
                            for _, row in ordered.iterrows()
                        )
                        lines.append(f"| {dataset} | {task} | {phase} | {ranking} |")

    protocol = manifest.get("protocol", {})
    lines += [
        "",
        "## Protocol",
        "",
        f"- Method: {manifest.get('config', {}).get('method', 'unknown')}",
        f"- Profile: {manifest.get('config', {}).get('profile', 'unknown')}",
        f"- Protocol version: {protocol.get('protocol_version', manifest.get('manifest_version', 1))}",
        f"- Split strategy: {protocol.get('split_strategy', 'legacy/unknown')}",
        f"- Preprocessing fit scope: {protocol.get('preprocessing_fit_scope', 'legacy/unknown')}",
        f"- Feature policy: {protocol.get('feature_policy', 'legacy/unknown')}",
        f"- SVM: maxIter={protocol.get('svm_max_iter', 'legacy/unknown')}, threshold={protocol.get('svm_threshold', 'legacy/unknown')}",
        f"- Minimal scaling: {protocol.get('minimal_scaling', 'legacy/unknown')}",
        "- Execution: local Spark on CPU; GPU and MapReduce values are not measured.",
        "",
        "## Collapsed classifiers",
        "",
    ]
    if collapsed.empty:
        lines.append("None detected.")
    else:
        for _, row in collapsed.iterrows():
            kval = "n/a" if pd.isna(row.get("k")) else int(row.k)
            lines.append(
                f"- {row.dataset} / {row.task} / {row.preprocessing} k={kval}: {row.collapse_reason}"
            )
        lines.append("\nCollapsed rows are retained for auditability and must not be treated as ordinary aggregate performance.")

    lines += ["", "## Published errata and ambiguities", ""]
    lines.extend(f"- {item}" for item in ERRATA)
    lines += [
        "",
        "## Timing limitations",
        "",
        "Published seconds are retained as context only. Absolute timing deltas are not scored because the hardware, sample sizes, Spark topology, and timing boundaries differ.",
        "",
        "## Artifacts",
        "",
        f"- Paper reference version: {REFERENCE_VERSION}",
        "- `paper_comparison.csv` contains every transcribed reference cell and its comparability status.",
        "- `model_diagnostics.jsonl` contains convergence evidence when produced by a version 2 run.",
    ]
    return "\n".join(lines) + "\n"


def compare_run(run_dir: Path) -> list[Path]:
    import pandas as pd

    run_dir = run_dir.resolve()
    frame = pd.DataFrame(comparison_rows(run_dir))
    csv_path = run_dir / "paper_comparison.csv"
    audit_path = run_dir / "REPRODUCTION_AUDIT.md"
    summary_path = run_dir / "SUMMARY.md"
    frame.to_csv(csv_path, index=False)
    audit = _audit_markdown(run_dir, frame)
    audit_path.write_text(audit, encoding="utf-8")
    summary_path.write_text(audit, encoding="utf-8")
    return [csv_path, audit_path, summary_path]
