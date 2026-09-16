"""Deterministic CSV-to-figure reporting."""

from __future__ import annotations

import json
from pathlib import Path


def _save_figure(figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def build_report(run_dir: Path) -> list[Path]:
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zeek-repro-matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from .comparison import compare_run, comparison_rows

    run_dir = run_dir.resolve()
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    metrics = _read_csv(run_dir / "metrics.csv")
    timings = _read_csv(run_dir / "timings.csv")
    variance = _read_csv(run_dir / "variance.csv")
    inventory = json.loads((run_dir / "inventory.json").read_text(encoding="utf-8"))
    sns.set_theme(style="whitegrid", context="notebook")
    written: list[Path] = []

    for item in inventory:
        counts = pd.Series(item["counts"], dtype=float).sort_values()
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(counts) * 0.38)))
        counts.plot.barh(ax=ax, color="#4472C4")
        ax.set(title=f"{item['dataset']}: MITRE ATT&CK tactic counts", xlabel="Records", ylabel="")
        ax.set_xscale("log")
        base = figures / f"{item['dataset']}-tactic-counts"
        _save_figure(fig, base)
        plt.close(fig)
        written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))

    if not metrics.empty:
        metrics["k"] = pd.to_numeric(metrics["k"], errors="coerce")
        for dataset, frame in metrics.groupby("dataset"):
            collapse_mask = (
                frame["collapse_flag"].astype(str).str.lower().eq("true")
                if "collapse_flag" in frame else pd.Series(False, index=frame.index)
            )
            headline = frame[~collapse_mask]
            comparison = headline.groupby(["task", "preprocessing"], as_index=False)["accuracy"].max()
            if comparison["preprocessing"].nunique() > 1:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                sns.barplot(comparison, x="task", y="accuracy", hue="preprocessing", ax=ax)
                excluded = int(collapse_mask.sum())
                suffix = f" (collapsed excluded: {excluded})" if excluded else ""
                ax.set(title=f"{dataset}: accuracy by preprocessing{suffix}", xlabel="Task", ylabel="Accuracy", ylim=(0, 1.02))
                ax.tick_params(axis="x", rotation=25)
                base = figures / f"{dataset}-accuracy-comparison"
                _save_figure(fig, base)
                plt.close(fig)
                written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))
            for method in ("pca", "lda"):
                subset = frame[frame["preprocessing"] == method].dropna(subset=["k"])
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5.5))
                sns.lineplot(subset, x="k", y="accuracy", hue="task", marker="o", ax=ax)
                if "collapse_flag" in subset:
                    collapsed = subset[subset["collapse_flag"].astype(str).str.lower().eq("true")]
                    if not collapsed.empty:
                        ax.scatter(
                            collapsed["k"], collapsed["accuracy"], marker="x", s=120,
                            linewidths=2.5, color="#C00000", label="collapsed classifier",
                        )
                ax.set(title=f"{dataset}: {method.upper()} SVM accuracy", xlabel="Components", ylabel="Accuracy", ylim=(0, 1.02))
                base = figures / f"{dataset}-{method}-accuracy"
                _save_figure(fig, base)
                plt.close(fig)
                written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))

        comparison = pd.DataFrame(comparison_rows(run_dir))
        accuracy = comparison[
            (comparison["metric"] == "accuracy")
            & comparison["comparison_status"].isin(("comparable", "collapsed-run-result"))
        ].copy()
        for dataset, frame in accuracy.groupby("dataset"):
            frame["identity"] = frame.apply(
                lambda row: f"{row.task} / {row.preprocessing} / k={int(row.k) if pd.notna(row.k) else 'n/a'}",
                axis=1,
            )
            long = frame.melt(
                id_vars=["identity", "collapse_flag"],
                value_vars=["paper_value", "reproduced_value"],
                var_name="source",
                value_name="accuracy",
            )
            fig, ax = plt.subplots(figsize=(11, max(5.5, len(frame) * 0.28)))
            sns.scatterplot(long, x="accuracy", y="identity", hue="source", style="source", s=70, ax=ax)
            collapsed = frame[frame["collapse_flag"]]
            if not collapsed.empty:
                ax.scatter(
                    collapsed["reproduced_value"], collapsed["identity"], marker="x",
                    s=150, linewidths=2.5, color="#C00000", label="collapsed classifier",
                )
                ax.legend()
            ax.set(title=f"{dataset}: published vs. reproduced accuracy", xlabel="Accuracy", ylabel="")
            base = figures / f"{dataset}-paper-accuracy-comparison"
            _save_figure(fig, base)
            plt.close(fig)
            written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))

    if not variance.empty:
        variance["k"] = pd.to_numeric(variance["k"], errors="coerce")
        for (dataset, method), frame in variance.groupby(["dataset", "preprocessing"]):
            maximum_k = frame.groupby("task")["k"].transform("max")
            subset = frame[frame["k"] == maximum_k]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            sns.lineplot(subset, x="component", y="explained_variance", hue="task", marker="o", ax=axes[0])
            sns.lineplot(subset, x="component", y="cumulative_variance", hue="task", marker="o", ax=axes[1])
            axes[0].set(title="Explained variance", xlabel="Component", ylabel="Fraction")
            axes[1].set(title="Cumulative variance", xlabel="Component", ylabel="Fraction", ylim=(0, 1.02))
            base = figures / f"{dataset}-{method}-variance"
            _save_figure(fig, base)
            plt.close(fig)
            written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))

    if not timings.empty:
        aggregate = (
            timings.groupby(["dataset", "task", "preprocessing"], as_index=False)
            .agg(
                preprocessing_seconds=("preprocessing_seconds", "median"),
                training_seconds=("training_seconds", "median"),
                evaluation_seconds=("evaluation_seconds", "median"),
            )
        )
        for dataset, frame in aggregate.groupby("dataset"):
            long = frame.melt(
                id_vars=["task", "preprocessing"],
                value_vars=["preprocessing_seconds", "training_seconds", "evaluation_seconds"],
                var_name="phase",
                value_name="seconds",
            )
            for phase, subset in long.groupby("phase"):
                fig, ax = plt.subplots(figsize=(10, 5.5))
                sns.barplot(subset, x="task", y="seconds", hue="preprocessing", ax=ax)
                ax.set(title=f"{dataset}: local CPU {phase.replace('_', ' ')}", xlabel="Task", ylabel="Seconds")
                ax.tick_params(axis="x", rotation=25)
                base = figures / f"{dataset}-cpu-{phase}"
                _save_figure(fig, base)
                plt.close(fig)
                written.extend((base.with_suffix(".png"), base.with_suffix(".pdf")))

    compare_run(run_dir)
    return written
