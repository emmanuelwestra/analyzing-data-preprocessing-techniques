"""Versioned transcription of the usable values in paper Tables 5-22."""

from __future__ import annotations

REFERENCE_VERSION = 1
STAT_METRICS = ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "attack_fpr", "auroc")
FRACTION_METRICS = STAT_METRICS + ("explained_variance", "cumulative_variance")
TIMING_METRICS = (
    "preprocessing_seconds",
    "cpu_training_seconds",
    "gpu_training_seconds",
    "cpu_testing_seconds",
    "gpu_testing_seconds",
)


def _expand(table, page, dataset, preprocessing, rows, metrics, status="valid", note=""):
    output = []
    for task, k, values in rows:
        for metric, value in zip(metrics, values):
            cell_status = status
            cell_note = note
            if table == 6 and task == "Reconnaissance" and k == 5 and metric == "weighted_f1":
                cell_status = "excluded"
                cell_note = "Published F1=98.99% is incompatible with precision=98.01% and recall=97.99%."
            output.append(
                {
                    "reference_version": REFERENCE_VERSION,
                    "table": table,
                    "page": page,
                    "dataset": dataset,
                    "preprocessing": preprocessing,
                    "task": task,
                    "k": k,
                    "metric": metric,
                    "paper_value": value / 100 if metric in FRACTION_METRICS else value,
                    "unit": "fraction" if metric in FRACTION_METRICS else "seconds",
                    "status": cell_status,
                    "note": cell_note,
                }
            )
    return output


def _variance(table, page, dataset, preprocessing, rows):
    output = []
    for task, values in rows.items():
        cumulative = 0.0
        for component, value in enumerate(values, 1):
            cumulative += value
            output.extend(
                _expand(
                    table,
                    page,
                    dataset,
                    preprocessing,
                    [(task, component, (value, cumulative))],
                    ("explained_variance", "cumulative_variance"),
                )
            )
    return output


def reference_rows() -> list[dict]:
    d22 = "UWF-ZeekData22"
    fall = "UWF-ZeekDataFall22"
    rows: list[dict] = []
    rows += _expand(5, 14, d22, "minimal", [
        ("Reconnaissance", None, (99.93, 99.93, 99.93, 99.92, .14, 99.92)),
        ("Discovery", None, (99.99, 99.99, 99.99, 99.99, 0, 99.99)),
        ("Multinomial", None, (99.94, 99.94, 99.92, 99.94, .03, 99.93)),
    ], STAT_METRICS)
    rows += _expand(6, 14, d22, "pca", [
        ("Reconnaissance", 2, (97.90,97.90,97.90,97.90,2.55,97.90)),
        ("Reconnaissance", 3, (98.15,98.16,98.16,98.16,2.45,98.16)),
        ("Reconnaissance", 5, (97.99,98.01,97.99,98.99,3.06,97.99)),
        ("Reconnaissance",10, (99.04,99.04,99.04,99.04,1.14,99.04)),
        ("Discovery",2,(99.78,99.57,99.78,99.67,0,49.99)),
        ("Discovery",3,(99.98,99.57,99.98,99.67,0,49.99)),
        ("Discovery",5,(99.78,99.61,99.78,99.67,0,50.25)),
        ("Discovery",10,(99.98,99.97,99.98,99.97,0,95.14)),
        ("Multinomial",2,(97.56,97.36,97.57,97.46,2.96,97.57)),
        ("Multinomial",3,(97.84,97.64,97.84,97.734,2.66,97.84)),
        ("Multinomial",5,(97.46,97.25,97.43,97.32,3.78,97.43)),
        ("Multinomial",10,(98.64,98.42,98.64,98.539,1.51,98.66)),
    ], STAT_METRICS)
    rows += _variance(7, 15, d22, "pca", {
        "Reconnaissance": [26.16,20.96,10.08,6.95,6.46,5.21,4.56,3.82,3.41,2.80],
        "Discovery": [20.82,15.11,14.37,8.57,7.81,5.53,4.69,4.38,3.81,3.60],
        "Multinomial": [26.16,20.94,10.11,6.93,6.42,5.19,4.55,3.82,3.41,2.86],
    })
    rows += _expand(8, 15, d22, "lda", [
        ("Reconnaissance",1,(99.96,99.96,99.96,99.96,.07,99.96)),
        ("Discovery",1,(99.94,99.94,99.94,99.94,.06,99.07)),
        ("Multinomial",1,(99.76,99.55,99.76,99.65,.03,99.77)),
        ("Multinomial",2,(99.75,99.54,99.75,99.65,.03,99.77)),
    ], STAT_METRICS)
    rows += _variance(9, 15, d22, "lda", {
        "Reconnaissance": [100.0], "Discovery": [100.0], "Multinomial": [99.82, .18]
    })
    rows += _expand(10, 16, fall, "minimal", [
        ("Reconnaissance",1,(99.96,99.96,99.96,99.96,.07,99.96)),
        ("Discovery",1,(99.94,99.94,99.94,99.94,.06,99.07)),
        ("Multinomial",1,(99.76,99.55,99.76,99.65,.03,99.77)),
        ("Multinomial",2,(99.75,99.54,99.75,99.65,.03,99.77)),
    ], STAT_METRICS, status="excluded", note="Table 10 duplicates Table 8 and omits required Fall22 tasks.")
    rows += _expand(11, 17, fall, "pca", [
        ("Reconnaissance",2,(87.27,76.17,87.28,81.34,0,50)),
        ("Reconnaissance",3,(92.69,92.23,92.69,92.33,2.72,79.24)),
        ("Reconnaissance",5,(90.01,90.96,90.01,87.06,.02,60.79)),
        ("Reconnaissance",10,(99.98,99.98,99.98,99.98,.02,99.98)),
        ("Discovery",2,(97.44,98.35,97.44,97.70,2.68,98.65)),
        ("Discovery",3,(98.50,98.86,98.50,98.60,1.57,99.18)),
        ("Discovery",5,(99.55,99.59,99.55,99.56,.47,99.73)),
        ("Discovery",10,(99.99,99.99,99.99,99.99,.001,99.99)),
        ("Defense_Evasion",2,(99.14,98.28,99.14,98.71,0,50)),
        ("Defense_Evasion",3,(99.14,98.28,99.14,98.71,0,50)),
        ("Defense_Evasion",5,(100,100,100,100,0,100)),
        ("Defense_Evasion",10,(100,100,100,100,0,100)),
        ("Privilege_Escalation",2,(99.14,98.28,99.14,98.71,0,50)),
        ("Privilege_Escalation",3,(99.14,98.28,99.13,98.71,0,50)),
        ("Privilege_Escalation",5,(100,100,100,100,0,100)),
        ("Privilege_Escalation",10,(100,100,100,100,0,100)),
        ("Resource_Development",2,(98.23,98.30,98.23,98.23,.004,98.42)),
        ("Resource_Development",3,(99.64,99.64,99.64,99.64,.59,99.67)),
        ("Resource_Development",5,(99.95,99.95,99.945,99.95,.06,99.95)),
        ("Resource_Development",10,(99.93,99.93,99.93,99.93,.09,99.93)),
        ("Multinomial",2,(87.21,78,87.21,82.33,4.46,90.95)),
        ("Multinomial",3,(90.47,84.32,90.47,87.04,.43,91.55)),
        ("Multinomial",5,(91.31,90.34,91.31,87.98,.06,92.44)),
        ("Multinomial",10,(92.07,90.55,92.07,90.62,.09,93.54)),
    ], STAT_METRICS)
    rows += _variance(12, 17, fall, "pca", {
        "Reconnaissance":[32.86,21.86,8.45,7.06,5.56,3.98,3.54,3.34,2.94,2.39],
        "Discovery":[33.26,24.55,9.36,6.08,4.38,3.81,3.51,2.90,2.76,2.66],
        "Defense_Evasion":[33.37,20.93,8.83,7.52,5.31,4.09,3.67,2.99,2.90,2.83],
        "Privilege_Escalation":[33.38,20.94,8.83,7.52,5.31,4.09,3.67,2.99,2.90,2.83],
        "Resource_Development":[33.10,26.26,12.05,6.94,4.19,3.36,2.89,2.85,2.04,2.00],
        "Multinomial":[32.21,24.47,12.74,6.84,4.06,3.51,3.26,2.98,2.07,1.97],
    })
    rows += _expand(13, 18, fall, "lda", [
        ("Reconnaissance",1,(100,100,100,100,0,100)),
        ("Discovery",1,(100,100,100,100,0,100)),
        ("Defense_Evasion",1,(100,100,100,100,0,100)),
        ("Privilege_Escalation",1,(100,100,100,100,0,100)),
        ("Resource_Development",1,(100,100,100,100,0,100)),
        ("Multinomial",1,(89.44,81.11,89.44,84.78,0,100)),
        ("Multinomial",2,(96.68,94.36,96.68,95.34,.001,99.99)),
        ("Multinomial",3,(97.83,97.32,97.83,97.49,.005,99.99)),
        ("Multinomial",4,(97.83,97.33,97.83,97.49,.006,99.99)),
        ("Multinomial",5,(97.83,97.33,97.83,97.49,.008,99.99)),
    ], STAT_METRICS)
    rows += _variance(14, 18, fall, "lda", {
        "Reconnaissance":[100], "Discovery":[100], "Defense_Evasion":[100],
        "Privilege_Escalation":[100], "Resource_Development":[100],
        "Multinomial":[99.55,.41,.04,0,0],
    })

    rows += _expand(15, 19, d22, "minimal", [
        ("Reconnaissance",None,(47.85,110.04,58.25,.45,.31)),
        ("Discovery",None,(60.04,102.19,57.21,.46,.32)),
        ("Multinomial",None,(60.55,280.82,160.34,.60,.40)),
    ], TIMING_METRICS)
    rows += _expand(16, 19, d22, "pca", [
        ("Reconnaissance",2,(102.97,146.92,89.71,.39,.21)), ("Reconnaissance",3,(103.28,152.56,90.01,.39,.22)),
        ("Reconnaissance",5,(109.21,150.16,80.98,.48,.25)), ("Reconnaissance",10,(104.18,148.81,81.23,.37,.19)),
        ("Discovery",2,(109.30,320.41,178.76,.41,.21)), ("Discovery",3,(106.02,301.83,167.32,.41,.23)),
        ("Discovery",5,(104.18,285.68,154.21,.42,.22)), ("Discovery",10,(109.75,298.21,149.11,.38,.20)),
        ("Multinomial",2,(107.46,339.83,176.19,.47,.21)), ("Multinomial",3,(132.52,403.01,210.98,.54,.31)),
        ("Multinomial",5,(102.79,271.49,145.21,.50,.32)), ("Multinomial",10,(104.05,282.63,153.24,.53,.31)),
    ], TIMING_METRICS)
    rows += _expand(17, 20, d22, "lda", [
        ("Reconnaissance",1,(167.97,23.28,16.31,.40,.30)), ("Discovery",1,(89.10,18.65,12.33,.27,.21)),
        ("Multinomial",1,(117.83,30.59,18.29,.29,.23)), ("Multinomial",2,(118.82,23.83,13.97,.33,.26)),
    ], TIMING_METRICS)
    mr_metrics=("preprocessing_seconds","cpu_mr_training_seconds","gpu_mr_training_seconds","cpu_mr_testing_seconds","gpu_mr_testing_seconds")
    rows += _expand(18, 20, d22, "lda-mapreduce", [
        ("Reconnaissance",1,(134.34,16.03,12.01,.27,.16)), ("Discovery",1,(121.40,18.25,10.13,.29,.19)),
        ("Multinomial",1,(138.21,35.44,19.43,.37,.21)), ("Multinomial",2,(145.07,28.32,20.21,.42,.25)),
    ], mr_metrics)
    rows += _expand(19, 21, fall, "minimal", [
        ("Reconnaissance",None,(38.26,143.06,89.32,.50,.34)), ("Discovery",None,(34.75,74.46,43.31,.45,.43)),
        ("Defense_Evasion",None,(35.39,77.36,45.22,.60,.50)), ("Privilege_Escalation",None,(36.03,72.30,43.33,.49,.35)),
        ("Resource_Development",None,(31.99,183.30,101.78,.63,.43)), ("Multinomial",None,(40.91,734.60,450.57,1.01,.78)),
    ], TIMING_METRICS)
    rows += _expand(20, 22, fall, "pca", [
        ("Reconnaissance",2,(49.65,325.02,180.91,.30,.22)), ("Reconnaissance",3,(55.45,127.01,90.23,.39,.31)),
        ("Reconnaissance",5,(54.95,124.90,87.45,.38,.23)), ("Reconnaissance",10,(50.70,120.84,95.61,.41,.24)),
        ("Discovery",2,(58.28,157.41,90.87,.36,.20)), ("Discovery",3,(49.73,110.05,70.81,.28,.18)),
        ("Discovery",5,(52.52,104.06,68.90,.27,.18)), ("Discovery",10,(48.82,114.01,76.32,.31,.20)),
        ("Defense_Evasion",2,(45.52,142.23,83.44,.39,.24)), ("Defense_Evasion",3,(48.37,149.94,89.43,.44,.27)),
        ("Defense_Evasion",5,(54.23,132.18,94.55,.39,.25)), ("Defense_Evasion",10,(48.82,114.01,78.90,.31,.20)),
        ("Privilege_Escalation",2,(53.16,141.69,82.98,.31,.20)), ("Privilege_Escalation",3,(54.51,142.89,84.56,.39,.24)),
        ("Privilege_Escalation",5,(58.38,124.57,78.34,.38,.22)), ("Privilege_Escalation",10,(54.44,65.24,50.12,.32,.20)),
        ("Resource_Development",2,(73.87,217.72,149.54,.44,.31)), ("Resource_Development",3,(69.89,191.16,120.76,.40,.28)),
        ("Resource_Development",5,(69.69,202.51,156.32,.47,.33)), ("Resource_Development",10,(61.97,204.99,144.45,.36,.22)),
        ("Multinomial",2,(81.62,1630.58,1200.23,.96,.71)), ("Multinomial",3,(79.74,965.02,721.34,.76,.65)),
        ("Multinomial",5,(84.46,907.64,701.45,.77,.69)), ("Multinomial",10,(84.16,772.88,561.90,.58,.32)),
    ], TIMING_METRICS)
    rows += _expand(21, 22, fall, "lda", [
        ("Reconnaissance",1,(69.28,38.33,19.17,.25,.13)), ("Discovery",1,(66.82,13.05,6.52,.26,.13)),
        ("Defense_Evasion",1,(67.20,13.49,6.74,.27,.13)), ("Privilege_Escalation",1,(66.19,11.47,5.74,.26,.13)),
        ("Resource_Development",1,(91.94,12.19,6.10,.23,.11)), ("Multinomial",1,(85.97,86.02,43.01,.55,.27)),
        ("Multinomial",2,(204.73,61.29,30.64,.68,.34)), ("Multinomial",3,(97.92,52.68,26.34,.58,.29)),
        ("Multinomial",4,(99.87,48.29,24.14,.50,.25)),
    ], TIMING_METRICS)
    rows += _expand(22, 23, fall, "lda-mapreduce", [
        ("Reconnaissance",1,(90.21,38.29,19.15,.27,.14)), ("Discovery",1,(88.91,13.32,6.66,.26,.13)),
        ("Defense_Evasion",1,(88.80,12.72,6.36,.30,.15)), ("Privilege_Escalation",1,(85.48,12.37,6.19,.29,.15)),
        ("Resource_Development",1,(130.66,16.08,8.04,.32,.16)), ("Multinomial",1,(137.42,97.20,48.60,.54,.27)),
        ("Multinomial",2,(147.35,87.71,43.85,.46,.23)), ("Multinomial",3,(163.40,83.05,41.53,.61,.30)),
        ("Multinomial",4,(177.51,106.20,53.10,.51,.25)),
    ], mr_metrics)
    return rows


ERRATA = [
    "Table 10 appears to duplicate Table 8 and is excluded from scoring.",
    "Table 6 Reconnaissance PCA k=5 reports an impossible F1 value and that cell is excluded.",
    "The methods mention PCA k=11, while the result tables stop at k=10.",
    "The text says Fall22 LDA tested k=1-4, while Table 13 includes k=5.",
    "The paper describes 0-1 scaling but gives the z-score equation used by Spark StandardScaler.",
    "Figure 9 and the Figure 10 discussion contain dataset-name caption inconsistencies.",
]
