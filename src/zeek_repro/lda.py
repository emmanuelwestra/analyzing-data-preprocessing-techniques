"""In-memory and sufficient-statistics LDA transformations."""

from __future__ import annotations

from time import perf_counter


def _transform_with_projection(df, mean, projection):
    import numpy as np
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql import functions as F

    mean = np.asarray(mean, dtype=float)
    projection = np.asarray(projection, dtype=float)
    broadcast_mean = df.sparkSession.sparkContext.broadcast(mean)
    broadcast_projection = df.sparkSession.sparkContext.broadcast(projection)

    @F.udf(VectorUDT())
    def project(vector):
        values = vector.toArray().astype(float)
        reduced = (values - broadcast_mean.value) @ broadcast_projection.value
        return Vectors.dense(reduced)

    return (
        df.withColumn("__reduced_features", project(F.col("features")))
        .drop("features")
        .withColumnRenamed("__reduced_features", "features")
    )


def sklearn_lda(train, test, k: int, fit_on_all: bool = False):
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    started = perf_counter()
    fit_frame = train.unionByName(test) if fit_on_all else train
    rows = fit_frame.select("label", "features").collect()
    x_train = np.vstack([row.features.toArray() for row in rows])
    y_train = np.asarray([row.label for row in rows])
    model = LinearDiscriminantAnalysis(solver="svd", n_components=k)
    model.fit(x_train, y_train)
    projection = model.scalings_[:, :k]
    mean = model.xbar_
    transformed_train = _transform_with_projection(train, mean, projection)
    transformed_test = _transform_with_projection(test, mean, projection)
    transformed_train.cache().count()
    transformed_test.cache().count()
    variance = [float(value) for value in model.explained_variance_ratio_[:k]]
    return transformed_train, transformed_test, variance, perf_counter() - started


def sufficient_statistics_lda(train, test, k: int, fit_on_all: bool = False):
    """Fit LDA using only per-class count, sum, and cross-product matrices."""
    import numpy as np

    started = perf_counter()
    fit_frame = train.unionByName(test) if fit_on_all else train
    dimension = fit_frame.select("features").first().features.size

    def sequence(accumulator, item):
        label, values = item
        vector = np.asarray(values, dtype=float)
        count, total, cross = accumulator.get(
            label, (0, np.zeros(dimension), np.zeros((dimension, dimension)))
        )
        accumulator[label] = (count + 1, total + vector, cross + np.outer(vector, vector))
        return accumulator

    def combine(left, right):
        for label, (count, total, cross) in right.items():
            l_count, l_total, l_cross = left.get(
                label, (0, np.zeros(dimension), np.zeros((dimension, dimension)))
            )
            left[label] = (l_count + count, l_total + total, l_cross + cross)
        return left

    stats = (
        fit_frame.select("label", "features")
        .rdd.map(lambda row: (float(row.label), row.features.toArray()))
        .treeAggregate({}, sequence, combine, depth=3)
    )
    total_count = sum(item[0] for item in stats.values())
    global_mean = sum(item[1] for item in stats.values()) / total_count
    within = np.zeros((dimension, dimension))
    between = np.zeros((dimension, dimension))
    for count, total, cross in stats.values():
        class_mean = total / count
        within += cross - np.outer(total, total) / count
        delta = class_mean - global_mean
        between += count * np.outer(delta, delta)
    matrix = np.linalg.pinv(within, hermitian=True) @ between
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    order = np.argsort(eigenvalues.real)[::-1]
    positive = np.maximum(eigenvalues.real[order], 0.0)
    projection = eigenvectors.real[:, order[:k]]
    projection /= np.maximum(np.linalg.norm(projection, axis=0), np.finfo(float).eps)
    denominator = positive.sum()
    variance = (positive[:k] / denominator).tolist() if denominator else [0.0] * k
    transformed_train = _transform_with_projection(train, global_mean, projection)
    transformed_test = _transform_with_projection(test, global_mean, projection)
    transformed_train.cache().count()
    transformed_test.cache().count()
    return transformed_train, transformed_test, variance, perf_counter() - started
