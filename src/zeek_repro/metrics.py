"""Metrics and confusion-matrix extraction shared by all stages."""

from __future__ import annotations

import math


def metrics_from_confusion(matrix, labels, auroc=math.nan):
    import numpy as np

    matrix = np.asarray(matrix, dtype=int)
    total = int(matrix.sum())
    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        per_class.append(
            {
                "label": float(label),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "support": int(matrix[index, :].sum()),
                "predicted_support": int(matrix[:, index].sum()),
            }
        )
    denominator = max(total, 1)
    weighted = lambda key: sum(row[key] * row["support"] for row in per_class) / denominator
    macro = lambda key: sum(row[key] for row in per_class) / max(len(per_class), 1)
    attack = per_class[labels.index(1.0)] if 1.0 in labels else {}
    collapsed_labels = [
        row["label"]
        for row in per_class
        if row["support"] > 0 and (row["predicted_support"] == 0 or row["recall"] == 0.0)
    ]
    missing_test_labels = [row["label"] for row in per_class if row["support"] == 0]
    attack_fpr = per_class[labels.index(1.0)]["fpr"] if 1.0 in labels else math.nan
    summary = {
        "accuracy": float(np.trace(matrix) / denominator),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_f1": weighted("f1"),
        "weighted_fpr": weighted("fpr"),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "auroc": float(auroc),
        "attack_precision": attack.get("precision", math.nan),
        "attack_recall": attack.get("recall", math.nan),
        "attack_f1": attack.get("f1", math.nan),
        "attack_fpr": attack_fpr,
        "per_class_fpr_json": __import__("json").dumps(
            {str(row["label"]): row["fpr"] for row in per_class}, sort_keys=True
        ),
        "predicted_support_json": __import__("json").dumps(
            {str(row["label"]): row["predicted_support"] for row in per_class}, sort_keys=True
        ),
        "collapse_flag": bool(collapsed_labels),
        "collapse_reason": (
            "zero predictions or recall for supported labels: "
            + ", ".join(str(label) for label in collapsed_labels)
            if collapsed_labels
            else ""
        ),
        "evaluation_valid": not missing_test_labels,
        "evaluation_warning": (
            "no test support for labels: " + ", ".join(str(label) for label in missing_test_labels)
            if missing_test_labels
            else ""
        ),
        "test_rows": total,
    }
    return summary, per_class, matrix.tolist()


def metrics_from_arrays(y_true, y_pred, scores=None, labels=None) -> tuple[dict, list[dict], list[list[int]]]:
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels if labels is not None else sorted(set(y_true) | set(y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    auroc = math.nan
    if scores is not None and len(set(y_true)) > 1:
        score_array = np.asarray(scores)
        try:
            if len(labels) == 2:
                positive_scores = score_array[:, -1] if score_array.ndim == 2 else score_array
                auroc = float(roc_auc_score(y_true, positive_scores))
            else:
                from sklearn.preprocessing import label_binarize

                auroc = float(
                    roc_auc_score(
                        label_binarize(y_true, classes=labels),
                        score_array,
                        average="weighted",
                    )
                )
        except ValueError:
            pass
    return metrics_from_confusion(matrix, labels, auroc=auroc)


def evaluate_spark_predictions(predictions, labels, distributed: bool = False):
    if distributed:
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F

        matrix = [[0 for _ in labels] for _ in labels]
        positions = {float(label): index for index, label in enumerate(labels)}
        for row in predictions.groupBy("label", "prediction").count().collect():
            matrix[positions[float(row.label)]][positions[float(row.prediction)]] = int(row["count"])
        try:
            if len(labels) == 2:
                auroc = BinaryClassificationEvaluator(
                    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
                ).evaluate(predictions)
            else:
                scored = predictions.withColumn("__scores", vector_to_array("rawPrediction"))
                supports = [sum(matrix[index]) for index in range(len(labels))]
                class_aurocs = []
                for index, label in enumerate(labels):
                    one_class = scored.withColumn(
                        "__binary_label", (F.col("label") == float(label)).cast("double")
                    ).withColumn("__score", F.col("__scores")[index])
                    class_aurocs.append(
                        BinaryClassificationEvaluator(
                            labelCol="__binary_label",
                            rawPredictionCol="__score",
                            metricName="areaUnderROC",
                        ).evaluate(one_class)
                    )
                auroc = sum(value * support for value, support in zip(class_aurocs, supports)) / sum(supports)
        except Exception:
            auroc = math.nan
        return metrics_from_confusion(matrix, labels, auroc=auroc)

    rows = predictions.select("label", "prediction", "rawPrediction").collect()
    y_true = [float(row.label) for row in rows]
    y_pred = [float(row.prediction) for row in rows]
    scores = []
    for row in rows:
        raw = row.rawPrediction
        scores.append(list(raw.toArray()) if hasattr(raw, "toArray") else float(raw))
    return metrics_from_arrays(y_true, y_pred, scores=scores, labels=labels)
