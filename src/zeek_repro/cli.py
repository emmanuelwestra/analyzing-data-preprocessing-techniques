"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATASETS, RunConfig
from .inventory import inventory


def _datasets(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values or DATASETS.keys())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="zeek-repro", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inventory_parser = commands.add_parser("inventory", help="validate the paper datasets")
    inventory_parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    inventory_parser.add_argument("--dataset", action="append", choices=DATASETS)

    run_parser = commands.add_parser("run", help="run one or all experiment stages")
    run_parser.add_argument("--stage", choices=("svm", "pca", "lda", "all"), required=True)
    run_parser.add_argument("--profile", choices=("development", "full"), default="development")
    run_parser.add_argument("--method", choices=("corrected", "paper-compatible"), default="corrected")
    run_parser.add_argument("--dataset", action="append", choices=DATASETS)
    run_parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    run_parser.add_argument("--output-root", type=Path, default=Path("results"))
    run_parser.add_argument("--cap", type=int, default=100_000)
    run_parser.add_argument("--repetitions", type=int)
    run_parser.add_argument("--no-warmup", action="store_true")
    run_parser.add_argument(
        "--feature-policy",
        choices=("corrected", "hashed-ips", "legacy-ordinal"),
    )
    run_parser.add_argument("--svm-max-iter", type=int)
    run_parser.add_argument("--svm-threshold", type=float)
    run_parser.add_argument("--minimal-scaling", choices=("scaled", "unscaled"))
    run_parser.add_argument("--diagnostic-sweep", action="store_true")
    run_parser.add_argument("--sensitivity-run", action="store_true")

    report_parser = commands.add_parser("report", help="rebuild figures from a completed run")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="compare a run with published values")
    compare_parser.add_argument("--run-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == "inventory":
        result = inventory(args.data_root, _datasets(args.dataset))
        print(json.dumps(result, indent=2))
        if any(item["problems"] for item in result):
            raise SystemExit(1)
        return
    if args.command == "report":
        from .reporting import build_report

        figures = build_report(args.run_dir)
        print(f"Wrote {len(figures)} figure files to {args.run_dir / 'figures'}")
        return
    if args.command == "compare":
        from .comparison import compare_run

        outputs = compare_run(args.run_dir)
        print("\n".join(f"Wrote {path}" for path in outputs))
        return

    repetitions = args.repetitions if args.repetitions is not None else (3 if args.profile == "development" else 1)
    if repetitions < 1 or args.cap < 1 or (args.svm_max_iter is not None and args.svm_max_iter < 1):
        raise SystemExit("--repetitions and --cap must be positive")
    if args.method == "paper-compatible" and not args.sensitivity_run:
        conflicts = []
        if args.feature_policy not in (None, "legacy-ordinal"):
            conflicts.append("--feature-policy")
        if args.svm_max_iter not in (None, 10):
            conflicts.append("--svm-max-iter")
        if args.svm_threshold not in (None, 0.00001):
            conflicts.append("--svm-threshold")
        if args.minimal_scaling not in (None, "scaled"):
            conflicts.append("--minimal-scaling")
        if conflicts:
            argument_parser.error(
                "paper-compatible overrides require --sensitivity-run: " + ", ".join(conflicts)
            )
    from .data import require_java_17

    try:
        require_java_17()
    except RuntimeError as exc:
        argument_parser.error(str(exc))
    if args.profile == "full":
        print("WARNING: full UWF-ZeekData22 runs can take hours and may exceed 16 GB RAM.")
    config = RunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        stage=args.stage,
        profile=args.profile,
        method=args.method,
        datasets=_datasets(args.dataset),
        development_cap=args.cap,
        repetitions=repetitions,
        warmup=not args.no_warmup,
        feature_policy=args.feature_policy,
        svm_max_iter=args.svm_max_iter,
        svm_threshold=args.svm_threshold,
        minimal_scaling=args.minimal_scaling,
        diagnostic_sweep=args.diagnostic_sweep,
        sensitivity_run=args.sensitivity_run,
    )
    from .reporting import build_report
    from .runner import run_experiments

    run_dir = run_experiments(config)
    build_report(run_dir)
    print(f"Completed run: {run_dir}")
