"""Spark loading, task construction, sampling, and splitting."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import DATASETS, PAPER_REDUCTIONS


@dataclass(frozen=True)
class ClassificationTask:
    name: str
    kind: str
    tactics: tuple[str, ...]
    label_mapping: dict[str, float]


def require_java_17() -> None:
    """Fail clearly before Spark starts with an unsupported Java runtime."""
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin" / "java") if java_home else "java"
    try:
        result = subprocess.run(
            [java, "-version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise RuntimeError(
            "JDK 17 is required to run PySpark 3.5.8. Install it and set JAVA_HOME."
        ) from exc

    version_output = result.stderr or result.stdout
    match = re.search(r'version "(?:1\.)?(\d+)', version_output)
    major = int(match.group(1)) if match else None
    if result.returncode != 0 or major != 17:
        detected = f"Java {major}" if major is not None else "an unknown Java version"
        raise RuntimeError(
            f"JDK 17 is required to run PySpark 3.5.8; detected {detected}. "
            "Set JAVA_HOME to a JDK 17 installation before running zeek-repro."
        )


def create_spark(app_name: str = "zeek-preprocessing-reproduction"):
    import sys

    from pyspark.sql import SparkSession

    require_java_17()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "10g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_dataset(spark, data_root: Path, dataset_name: str):
    directory = DATASETS[dataset_name]["directory"]
    return (
        spark.read.option("recursiveFileLookup", "true")
        .parquet(str(data_root / directory / "parquet"))
    )


def tasks_for(dataset_name: str) -> tuple[ClassificationTask, ...]:
    spec = DATASETS[dataset_name]
    binary = tuple(
        ClassificationTask(
            name=tactic.replace(" ", "_"),
            kind="binary",
            tactics=(tactic,),
            label_mapping={"none": 0.0, tactic: 1.0},
        )
        for tactic in spec["binary_tactics"]
    )
    multi_tactics = spec["multinomial_tactics"]
    mapping = {"none": 0.0, **{t: float(i + 1) for i, t in enumerate(multi_tactics)}}
    return binary + (
        ClassificationTask(
            name="Multinomial",
            kind="multinomial",
            tactics=multi_tactics,
            label_mapping=mapping,
        ),
    )


def build_task_frame(df, task: ClassificationTask):
    from pyspark.sql import functions as F

    allowed = ("none",) + task.tactics
    mapping_items = []
    for key, value in task.label_mapping.items():
        mapping_items.extend((F.lit(key), F.lit(value)))
    mapping = F.create_map(*mapping_items)
    return df.filter(F.col("label_tactic").isin(*allowed)).withColumn(
        "label", mapping[F.col("label_tactic")].cast("double")
    )


def deterministic_class_cap(df, cap: int, seed: int):
    """Keep the lowest stable uid hashes per class, exactly up to ``cap``."""
    from pyspark.sql import Window, functions as F

    stable_key = F.coalesce(F.col("uid"), F.concat_ws("|", *df.columns))
    window = Window.partitionBy("label_tactic").orderBy(
        F.xxhash64(stable_key, F.lit(seed)), stable_key
    )
    return (
        df.withColumn("__sample_rank", F.row_number().over(window))
        .filter(F.col("__sample_rank") <= cap)
        .drop("__sample_rank")
    )


def deterministic_paper_sample(df, stage: str, task: ClassificationTask, seed: int):
    """Reconstruct the paper's reported aggregate reductions deterministically.

    Discovery is always retained. Remaining capacity is allocated proportionally
    across the large classes, with stable hash ordering within each class.
    """
    from pyspark.sql import Window, functions as F

    reduction = PAPER_REDUCTIONS[stage][task.name]
    if reduction <= 0:
        return df, {
            "policy": "paper-reported-reduction",
            "reported_reduction": reduction,
            "target_rows": df.count(),
        }

    counts = {row.label_tactic: int(row["count"]) for row in df.groupBy("label_tactic").count().collect()}
    total = sum(counts.values())
    target_total = max(1, round(total * (1.0 - reduction)))
    protected = {"Discovery"} & set(counts)
    targets = {label: counts[label] for label in protected}
    remaining = max(0, target_total - sum(targets.values()))
    scalable = {label: count for label, count in counts.items() if label not in protected}
    scalable_total = sum(scalable.values())
    allocated = 0
    for label in sorted(scalable):
        target = min(scalable[label], int(remaining * scalable[label] / max(scalable_total, 1)))
        targets[label] = target
        allocated += target
    for label in sorted(scalable, key=lambda item: (-scalable[item], item)):
        if allocated >= remaining:
            break
        room = scalable[label] - targets[label]
        addition = min(room, remaining - allocated)
        targets[label] += addition
        allocated += addition

    stable_key = F.coalesce(F.col("uid"), F.concat_ws("|", *df.columns))
    window = Window.partitionBy("label_tactic").orderBy(
        F.xxhash64(stable_key, F.lit(seed)), stable_key
    )
    target_map = []
    for label, count in targets.items():
        target_map.extend((F.lit(label), F.lit(count)))
    sampled = (
        df.withColumn("__sample_rank", F.row_number().over(window))
        .filter(F.col("__sample_rank") <= F.create_map(*target_map)[F.col("label_tactic")])
        .drop("__sample_rank")
    )
    return sampled, {
        "policy": "paper-reported-reduction",
        "reported_reduction": reduction,
        "source_rows": total,
        "target_rows": sum(targets.values()),
        "target_class_counts": targets,
        "discovery_retained": counts.get("Discovery", 0) == targets.get("Discovery", 0),
    }


def sample_for_run(df, dataset_name, stage, task, profile, method, cap, seed):
    if profile == "development":
        sampled = deterministic_class_cap(df, cap, seed)
        return sampled, {"policy": "development-class-cap", "cap_per_class": cap}
    if method == "paper-compatible" and dataset_name == "UWF-ZeekData22":
        return deterministic_paper_sample(df, stage, task, seed)
    return df, {"policy": "full-eligible-data"}


def class_counts(df) -> dict[str, int]:
    return {row.label_tactic: int(row["count"]) for row in df.groupBy("label_tactic").count().collect()}


def stratified_split(df, seed: int, train_fraction: float = 0.7):
    """Deterministic per-class split using a stable uid hash."""
    from pyspark.sql import functions as F

    stable_key = F.coalesce(F.col("uid"), F.concat_ws("|", *df.columns))
    bucket = F.pmod(F.xxhash64(stable_key, F.lit(seed)), F.lit(10_000))
    cutoff = int(train_fraction * 10_000)
    return df.filter(bucket < cutoff), df.filter(bucket >= cutoff)


def paper_split(df, dataset_name: str, task: ClassificationTask, seed: int):
    if dataset_name == "UWF-ZeekData22" and task.name == "Reconnaissance":
        return stratified_split(df, seed)
    return tuple(df.randomSplit([0.7, 0.3], seed=seed))
