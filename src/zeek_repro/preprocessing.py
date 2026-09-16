"""Shared Spark preprocessing with auditable fitted metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreparedData:
    train: object
    test: object
    metadata: dict
    seconds: float


def _feature_frame(df, feature_policy: str):
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, NumericType, StringType, TimestampType

    dropped = {"uid", "label_tactic", "label", "label_binary", "label_technique"}
    if feature_policy == "corrected":
        dropped.update({"community_id", "src_ip_zeek", "dest_ip_zeek"})
    elif feature_policy == "hashed-ips":
        dropped.add("community_id")
    categorical: list[str] = []
    numeric: list[str] = []
    result = df
    for field in df.schema.fields:
        if field.name in dropped:
            continue
        output = f"__value_{field.name}"
        if isinstance(field.dataType, StringType):
            value = F.coalesce(F.col(field.name), F.lit(""))
            if feature_policy in {"corrected", "hashed-ips"}:
                value = F.when(F.trim(value) == "", F.lit("__MISSING__")).otherwise(value)
            result = result.withColumn(output, value)
            categorical.append(output)
        elif isinstance(field.dataType, TimestampType):
            result = result.withColumn(output, F.col(field.name).cast("double"))
            numeric.append(output)
        elif isinstance(field.dataType, (NumericType, BooleanType)):
            result = result.withColumn(output, F.col(field.name).cast("double"))
            numeric.append(output)
    return result, numeric, categorical


def fit_preprocessor(
    train,
    test,
    fit_on_all: bool = False,
    feature_policy: str = "legacy-ordinal",
    scale: bool = True,
) -> PreparedData:
    from time import perf_counter

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        FeatureHasher,
        Imputer,
        OneHotEncoder,
        StandardScaler,
        StringIndexer,
        VectorAssembler,
    )

    started = perf_counter()
    train_values, numeric, categorical = _feature_frame(train, feature_policy)
    test_values, _, _ = _feature_frame(test, feature_policy)
    fit_frame = train_values.unionByName(test_values) if fit_on_all else train_values

    numeric_imputed = [f"{name}__imputed" for name in numeric]
    hashed = [name for name in categorical if feature_policy == "hashed-ips" and name in {"__value_src_ip_zeek", "__value_dest_ip_zeek"}]
    indexed_inputs = [name for name in categorical if name not in hashed]
    indexed = [f"{name}__indexed" for name in indexed_inputs]
    stages = [
        StringIndexer(inputCol=name, outputCol=output, handleInvalid="keep")
        for name, output in zip(indexed_inputs, indexed)
    ]
    assembled_categorical = list(indexed)
    if feature_policy in {"corrected", "hashed-ips"} and indexed:
        encoded = [f"{name}__onehot" for name in indexed_inputs]
        stages.append(OneHotEncoder(inputCols=indexed, outputCols=encoded, handleInvalid="keep"))
        assembled_categorical = encoded
    hashed_output = None
    if hashed:
        hashed_output = "__hashed_ips"
        stages.append(
            FeatureHasher(
                inputCols=hashed,
                outputCol=hashed_output,
                numFeatures=256,
                categoricalCols=hashed,
            )
        )
    if numeric:
        stages.append(Imputer(inputCols=numeric, outputCols=numeric_imputed, strategy="mean"))
    assembler_inputs = numeric_imputed + assembled_categorical + ([hashed_output] if hashed_output else [])
    assembler_output = "unscaled_features" if scale else "features"
    stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol=assembler_output, handleInvalid="keep"))
    if scale:
        stages.append(
            StandardScaler(
                inputCol="unscaled_features",
                outputCol="features",
                withMean=True,
                withStd=True,
            )
        )
    model = Pipeline(stages=stages).fit(fit_frame)
    prepared_train = model.transform(train_values).select("label", "features", "label_tactic")
    prepared_test = model.transform(test_values).select("label", "features", "label_tactic")
    prepared_train.cache().count()
    prepared_test.cache().count()

    string_mappings = {}
    for stage in model.stages:
        if hasattr(stage, "labels") and hasattr(stage, "getInputCol"):
            string_mappings[stage.getInputCol().replace("__value_", "")] = list(stage.labels)
    unseen_category_counts = {}
    if not fit_on_all:
        for name in indexed_inputs:
            known_values = fit_frame.select(name).distinct()
            unseen_category_counts[name.replace("__value_", "")] = test_values.select(
                name
            ).join(known_values, on=name, how="left_anti").count()
    feature_dimension = prepared_train.select("features").first().features.size
    hash_metadata = {}
    if hashed:
        distinct = {
            name.replace("__value_", ""): fit_frame.select(name).distinct().count()
            for name in hashed
        }
        total_distinct = sum(distinct.values())
        hash_metadata = {
            "buckets": 256,
            "distinct_values": distinct,
            "minimum_collisions": max(0, total_distinct - 256),
            "load_factor": total_distinct / 256,
        }
    metadata = {
        "numeric_features": [name.replace("__value_", "") for name in numeric],
        "categorical_features": [name.replace("__value_", "") for name in categorical],
        "string_index_mappings": string_mappings,
        "unseen_test_categories": unseen_category_counts,
        "hashing": hash_metadata,
        "feature_policy": feature_policy,
        "feature_dimension": feature_dimension,
        "scaling": "z-score (withMean=True, withStd=True)" if scale else "none",
        "fit_scope": "train+test" if fit_on_all else "train-only",
    }
    return PreparedData(prepared_train, prepared_test, metadata, perf_counter() - started)


def apply_pca(train, test, k: int, fit_on_all: bool = False):
    from time import perf_counter

    from pyspark.ml.feature import PCA

    started = perf_counter()
    fit_frame = train.unionByName(test) if fit_on_all else train
    model = PCA(k=k, inputCol="features", outputCol="reduced_features").fit(fit_frame)
    output_train = model.transform(train).drop("features").withColumnRenamed(
        "reduced_features", "features"
    )
    output_test = model.transform(test).drop("features").withColumnRenamed(
        "reduced_features", "features"
    )
    output_train.cache().count()
    output_test.cache().count()
    variance = [float(value) for value in model.explainedVariance]
    return output_train, output_test, variance, perf_counter() - started
