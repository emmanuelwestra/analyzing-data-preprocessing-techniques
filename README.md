# Zeek preprocessing study reproduction

This repository reproduces, on a local CPU, the SVM experiments from Bagui et al.
(2025) using the original UWF-ZeekData22 and UWF-ZeekDataFall22 Parquet files. The
workflow is deliberately staged: establish the minimally preprocessed SVM baseline,
then add PCA, and finally LDA. Every run writes its configuration, software and
hardware metadata, metrics, timings, variance, confusion matrices, and figures.

The implementation does **not** claim to reproduce the paper's NVIDIA GPU or
distributed Hadoop/MapReduce timings. This repository targets an Apple Silicon Mac
and labels all timings as local CPU measurements.

The paper contains contradictory statements, apparent table and caption errors, and
method details that cannot be recovered from the publication. See
[`PAPER_INCONSISTENCIES.md`](PAPER_INCONSISTENCIES.md) for the complete audit and every
assumption made by this reproduction.

## Study data

Only these directories are used by default:

- `data/raw/UWF-ZeekData22` (8 Parquet files; 18,562,468 rows)
- `data/raw/UWF-ZeekDataFall22` (13 Parquet files; 700,340 rows)

The supplied 2024 and 2025 UWF datasets are intentionally excluded because they were
not used in the base paper. They can support a later external-validity study without
changing this reproduction's baseline.

Download every URL listed in `data/uwf_urls.txt` with GNU Wget:

```bash
wget --continue \
  --input-file=data/uwf_urls.txt \
  --directory-prefix=data/raw \
  --no-host-directories \
  --cut-dirs=1
```

Verify the exact paper datasets before modeling:

```bash
zeek-repro inventory
```

The command exits unsuccessfully if file totals, row totals, tactic counts, or schema
consistency differ from the expected 2022 data. A `full` experiment refuses to start
when this validation fails.

## Environment

The supported environment is:

- macOS on Apple Silicon (CPU only)
- Python 3.11
- JDK 17
- Apache Spark/PySpark 3.5.x

Spark 3.5 is not supported on the JDK 25 currently selected on some recent macOS
installations. Install a JDK 17 distribution and select it before running Spark. For
example, with Homebrew and `uv`:

```bash
brew install openjdk@17 uv
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

With standard `venv` and `pip` instead:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The versions in `pyproject.toml` are pinned so a completed run can be compared with a
later rerun. Raw data, virtual environments, and generated `results/` are ignored by
Git.

## Reproduction workflow

Development runs use seed 42 and retain up to 100,000 records from each eligible
class. Smaller classes are retained in full. This makes the complete workflow
practical on a 16 GB Mac while preserving rare tactics.

Run and inspect the baseline first:

```bash
zeek-repro run --stage svm --profile development --method corrected
```

Then run PCA and LDA independently:

```bash
zeek-repro run --stage pca --profile development --method corrected
zeek-repro run --stage lda --profile development --method corrected
```

To produce cross-method comparison figures in one result directory, run all stages:

```bash
zeek-repro run --stage all --profile development --method corrected
```

The two study tracks are intentionally separate. `paper-compatible` preserves the
published SVM settings and the empirically matching transform-before-split behavior;
`corrected` splits first, fits transforms on training data only, removes identifier-like
categoricals, one-hot encodes low-cardinality strings, and uses stronger convergence
defaults:

```bash
zeek-repro run --stage all --profile development --method paper-compatible
zeek-repro run --stage all --profile development --method corrected --diagnostic-sweep
```

Paper-compatible parameter or feature overrides require `--sensitivity-run`. The
paper's Figure 6 scaling interpretation can be tested explicitly with:

```bash
zeek-repro run --stage svm --method paper-compatible --minimal-scaling unscaled --sensitivity-run
```

Useful options include:

```bash
# Run just one paper dataset.
zeek-repro run --stage svm --dataset UWF-ZeekDataFall22

# Short smoke run without repeated timing measurements.
zeek-repro run --stage all --cap 2000 --repetitions 1 --no-warmup

# Rebuild figures without fitting models again.
zeek-repro report --run-dir results/<run-id>
zeek-repro compare --run-dir results/<run-id>
```

`compare` writes `paper_comparison.csv`, `REPRODUCTION_AUDIT.md`, and an expanded
`SUMMARY.md`. Published GPU and MapReduce timings remain reference-only until those
backends are actually measured.

A full-data run uses all eligible records:

```bash
zeek-repro run --stage svm --profile full --method corrected
```

For timing claims, use three recorded repetitions explicitly:

```bash
zeek-repro run --stage all --profile full --method corrected --repetitions 3
```

Later GPU or distributed implementations must populate the existing manifest fields
`execution_mode`, `accelerator`, and `distributed`; they must never copy published
timings into measured output columns.

Full UWF-ZeekData22 SVM/PCA runs may take hours and exceed the practical memory or
thermal limits of a 16 GB laptop. Development LDA uses scikit-learn's SVD solver, as
described in the paper. Full-data LDA instead uses memory-bounded class counts, sums,
and cross-product matrices; its manifest labels it `local-sufficient-statistics`, not
an exact reproduction of the paper's in-memory scikit-learn execution.

By default, development runs perform one unrecorded warm-up and three recorded model
fits. Full runs record one fit. Preprocessing is materialized once per task and its
time is recorded separately from each model fit and evaluation.

## Methods

### Tasks and model

UWF-ZeekData22 uses Reconnaissance and Discovery as separate attack-versus-benign
binary tasks, plus a three-class multinomial task. UWF-ZeekDataFall22 adds Defense
Evasion, Privilege Escalation, and Resource Development, plus a six-class multinomial
task. Unused rare tactics are filtered before sampling.

Paper-compatible runs use Spark `LinearSVC` with the published settings:
`maxIter=10`, `regParam=0.0`, intercept enabled, and threshold `0.00001`.
Corrected runs use `maxIter=100` and threshold `0.0` by default. Multinomial
classification wraps the estimator in Spark `OneVsRest`. A collapsed model is flagged
when a supported class receives no predictions or has zero recall; an optional
diagnostic sweep evaluates iterations 10/50/100/200 at thresholds 0.00001 and 0.0
without replacing the faithful result.

The paper-compatible feature pipeline uses the legacy ordinal string-indexing
behavior. The corrected primary policy drops `community_id` and source/destination IP,
then one-hot encodes the remaining categorical fields. `--feature-policy hashed-ips`
retains IP information through fixed 256-bucket feature hashing, while
`--feature-policy legacy-ordinal` provides an ablation against the old pipeline.
Fitted feature metadata, unseen-category counts, class counts, fit scope, and hash
load/collision statistics are saved in `manifest.json`.

PCA runs `k = 2, 3, 5, 10, 11`. Results through 10 match the paper's principal tables,
while 11 is supplemental because the methods section says it was tested. LDA runs the
maximum defensible range: one discriminant for binary tasks, 1–2 for the three-class
UWF-ZeekData22 task, and 1–5 for the six-class UWF-ZeekDataFall22 task.

### Corrected and paper-compatible profiles

`--method corrected` is the primary scientific result. It creates a deterministic,
stratified 70:30 split before fitting imputation, indexing, scaling, PCA, or LDA, so
test data cannot influence a learned transform.

`--method paper-compatible` is a sensitivity analysis. It follows the paper's
apparent transform-before-split flow and uses its special stratified Reconnaissance
split for UWF-ZeekData22, with seeded Spark random splits elsewhere. This profile is
an interpretation, not recovered original code.

The paper says both that values were scaled to 0–1 and that Spark `StandardScaler`
performed `(x - mean) / standard deviation`. Those statements are inconsistent. Both
profiles use the published equation: mean-centered z-score scaling. The manifest
records that decision.

### Metrics and timing

Each model records accuracy, weighted precision/recall/F1/FPR, macro
precision/recall/F1, AUROC where the model output supports it, and binary attack-class
precision/recall/F1/FPR. Multiclass rows include per-class FPR, weighted FPR, predicted
support, and collapse status. Confusion matrices contain per-class metrics and the
exact label mapping. Weighted metrics are retained for comparability with Spark-style
values in the paper; macro and attack-class metrics make severe class imbalance visible.

Timing uses `time.perf_counter()` and separates preprocessing, dimensionality
reduction, SVM fitting, prediction/materialization, and metric computation. Timings
are machine-specific and should be interpreted as directional ranks within one run,
not as direct replicas of the paper's Tesla T4 and Hadoop-cluster measurements.

## Outputs

Each invocation creates `results/<UTC-run-id>/` containing:

```text
manifest.json              run configuration, environment, feature mappings
inventory.json             input files, schemas, row and tactic counts
metrics.csv                one auditable summary row per model
timings.csv                individual recorded timing observations
variance.csv               PCA/LDA explained and cumulative variance
model_diagnostics.jsonl    SVM/One-vs-Rest iterations and objective histories
diagnostic_sweeps.csv      explicitly requested collapse-diagnostic candidates
paper_comparison.csv       all reference cells and comparability decisions
confusion_matrices/*.json  matrices, class mappings, and per-class metrics
figures/*.png              raster publication figures
figures/*.pdf              vector publication figures
REPRODUCTION_AUDIT.md       answer-first comparison, errata, and limitations
SUMMARY.md                 copy of the complete reproduction audit
```

Reports include tactic-count plots, accuracy comparisons, accuracy versus PCA/LDA
dimension, explained/cumulative variance, and local CPU preprocessing/training/testing
comparisons. Figures are always rebuilt from the CSV artifacts. No GPU or MapReduce
series is silently copied from the paper or labeled as measured here.

## Verification

Run the fast unit tests with:

```bash
pytest -m 'not integration'
```

Run Spark integration tests when JDK 17 is active:

```bash
pytest -m integration
```

Run the capped acceptance matrix for both protocols with:

```bash
zeek-repro run --stage all --profile development --method paper-compatible --cap 2000 --repetitions 1 --no-warmup
zeek-repro run --stage all --profile development --method corrected --cap 2000 --repetitions 1 --no-warmup --diagnostic-sweep
pytest -m 'not integration'
pytest -m integration
```

Then run full-data CPU benchmarking separately, with one fit per configuration unless
the result will support a timing claim:

```bash
zeek-repro run --stage all --profile full --method paper-compatible --repetitions 1 --no-warmup
zeek-repro run --stage all --profile full --method corrected --repetitions 1 --no-warmup
```

Exact equality with the paper is not expected because its source code and important
protocol details were not published. The complete list is maintained in
[`PAPER_INCONSISTENCIES.md`](PAPER_INCONSISTENCIES.md). The comparison report quantifies
every match, exclusion, missing result, and hardware-limited timing instead of silently
resolving ambiguities in whichever direction scores better.

## Base paper

Bagui, S. S., Eller, C., Armour, R., Singh, S., Bagui, S. C., & Mink, D. (2025).
“Analyzing Performance of Data Preprocessing Techniques on CPUs vs. GPUs with and
Without the MapReduce Environment.” *Electronics, 14*(18), 3597.
https://doi.org/10.3390/electronics14183597

```bibtex
@article{bagui2025preprocessing,
  author  = {Bagui, Sikha S. and Eller, Colin and Armour, Rianna and Singh, Shivani and Bagui, Subhash C. and Mink, Dustin},
  title   = {Analyzing Performance of Data Preprocessing Techniques on {CPU}s vs. {GPU}s with and Without the {MapReduce} Environment},
  journal = {Electronics},
  year    = {2025},
  volume  = {14},
  number  = {18},
  pages   = {3597},
  doi     = {10.3390/electronics14183597}
}
```
