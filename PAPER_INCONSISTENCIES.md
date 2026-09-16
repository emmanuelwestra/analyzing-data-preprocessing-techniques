# Paper inconsistencies and reproduction assumptions

This document is the repository's complete audit of inconsistencies, apparent errors,
underspecified methods, and implementation assumptions in Bagui et al. (2025),
"Analyzing Performance of Data Preprocessing Techniques on CPUs vs. GPUs with and
Without the MapReduce Environment."

The list distinguishes errors in the paper from choices that a reproduction must make
because the source code, exact samples, or protocol details were not published. A
"paper-compatible" run should therefore be understood as a documented interpretation,
not an exact recovery of the authors' original implementation.

## Direct inconsistencies and apparent errors

### 1. Scaling is described as both min-max normalization and z-score standardization

Section 4.2.2 says that numerical features were scaled to a range between 0 and 1 and
calls Spark `StandardScaler` a min-max scaler. The displayed equation is instead

```text
x' = (x - mean) / standard deviation
```

which is z-score standardization. Spark `StandardScaler` performs standard-deviation
scaling and optionally mean centering; it is not a min-max scaler and does not bound
values to `[0, 1]`.

The reproduction uses mean-centered z-score scaling because it matches the published
equation. This is a choice between contradictory descriptions.

### 2. The role of scaling in "minimal preprocessing" is inconsistent

Section 7 says normalization occurs in cases where PCA or LDA feature reduction is
used, which implies that the minimal branch is not normalized. Section 8.2 later lists
normalization among the steps performed for minimal preprocessing. Consequently, the
paper does not establish whether the baseline SVM received scaled or unscaled numeric
features.

The paper-compatible default in this repository scales the minimal branch. An
unscaled sensitivity run is available through `--minimal-scaling unscaled
--sensitivity-run`.

### 3. The explanation of scaling and outliers is inaccurate

The paper says its normalization reduces sensitivity to outliers. Mean and standard
deviation are themselves sensitive to outliers, and z-score standardization does not
bound extreme values. Scaling may prevent units or magnitudes from dominating, but it
does not provide robust outlier handling.

### 4. UWF-ZeekData22 Discovery sampling contradicts itself

Section 7 says all Discovery instances were retained because the class was small.
Table 6 reports a 10% reduction for the Discovery PCA experiments, and Table 8 reports
a 5% reduction for the Discovery LDA experiment. Thus the prose and result tables
cannot all describe the same sampling policy.

### 5. UWF-ZeekData22 sample reductions are not sufficiently specified

The paper reports task-level percentage reductions and says reductions varied across
benign, Reconnaissance, and Discovery records. It does not give the final per-class
counts, selected row identifiers, or selection algorithm. Many different datasets can
satisfy the published percentages, particularly for the highly imbalanced Discovery
experiments.

### 6. Table 6 contains an impossible Reconnaissance F1 value

For Reconnaissance PCA at `k=5`, Table 6 reports precision 98.01%, recall 97.99%, and
F1 98.99%. An F1 score cannot exceed both its precision and recall. The F1 cell is
excluded from comparisons.

### 7. PCA `k=11` is claimed but no `k=11` result is reported

Section 8.2 says PCA was run with `k = 2, 3, 5, 10, 11` on both datasets. The PCA
classification and timing tables stop at `k=10`; no `k=11` statistical, timing, or
variance result is supplied.

This repository runs `k=11` as a supplemental configuration but does not treat it as a
published comparison row.

### 8. Fall22 LDA dimensions conflict across the text and Table 13

Section 8.2 says Fall22 multinomial LDA tested `k = 1, 2, 3, 4`, while Table 13 also
contains `k=5`. In addition, the paragraph introducing Table 13 says the LDA results
used `k = 2, 3, 5, 10`, even though the table actually contains binary `k=1` and
multinomial `k=1,2,3,4,5`. The introductory sentence appears to have been copied from
the PCA description.

The reproduction evaluates the mathematically permissible range: one discriminant for
binary tasks and `k=1` through `5` for the six-class Fall22 task.

### 9. Table 10 is mislabeled or duplicated

Table 10 is titled as Fall22 SVM with minimal preprocessing, but it contains only
Reconnaissance, Discovery, and two multinomial rows at `k=1` and `k=2`. Its values
duplicate Table 8, the UWF-ZeekData22 LDA results. It omits the additional Fall22
binary tactics and gives dimensions that a minimal-preprocessing table should not
have.

Table 10 is excluded from scoring because the intended Fall22 baseline values cannot
be recovered from it.

### 10. The Figure 8 discussion cites the wrong results table

The Fall22 PCA discussion below Figure 8 says that other statistical measures can be
seen in Table 9. Table 9 contains UWF-ZeekData22 LDA explained variance. The relevant
Fall22 PCA classification results are in Table 11.

### 11. Figure 9 has the wrong dataset in its caption

The surrounding section and discussion describe Figure 9 as the UWF-ZeekData22 timing
comparison. Its caption identifies UWF-ZeekDataFall22.

### 12. The Figure 10 discussion names the wrong dataset

Figure 10 and its surrounding section concern UWF-ZeekDataFall22, but the discussion
says that Figure 10 compares results for UWF-ZeekData22.

### 13. Figure 7 does not define how unlike result sets were aggregated

Figure 7 compares minimal preprocessing, PCA, and LDA accuracy. Minimal preprocessing
has one result per task, while PCA and multinomial LDA have multiple results across
`k`. The paper does not state precisely which values or averaging rule generated the
PCA and LDA bars. The chart therefore cannot be reconstructed unambiguously from its
caption and methods alone.

## Unspecified or ambiguous implementation details

### 14. Preprocessing fit scope and data leakage are not stated

The paper does not unambiguously say whether mean imputation, string indexing,
standardization, PCA, and supervised LDA were fitted on training data only or on the
complete dataset before the 70:30 split. Figure 6 and the ordering of the prose suggest
a transform-before-split flow, but do not establish it conclusively.

The paper-compatible profile fits these transformations on combined train and test
data to follow that apparent flow. The corrected profile splits first and fits all
learned transformations on training data only. The former can leak test-distribution
and, for LDA, test-label information.

### 15. Most split procedures are not defined

The paper specifies a 70:30 split, seed 42, stratified sampling for UWF-ZeekData22
Reconnaissance, and random splitting for Fall22. It does not define the split method
for UWF-ZeekData22 Discovery or multinomial classification, whether splits are exactly
or approximately 70:30, or whether the same records are reused across preprocessing
methods.

The paper-compatible implementation uses a deterministic stratified split for the
specified Reconnaissance task and seeded Spark `randomSplit` elsewhere. That is an
implementation assumption.

### 16. The exact reduced UWF-ZeekData22 records cannot be recovered

No row identifiers or random-sampling procedure are published for the 85-97%
reductions. This repository reconstructs the reported aggregate reductions
deterministically: it derives class targets from each task-level percentage and selects
records by stable seeded UID hashes. Discovery is protected when possible. These are
new reproducible samples, not the authors' unrecoverable original samples.

### 17. String indexing is underspecified

The paper says `StringIndexer` maps strings to numerical values, but does not state:

- the exact columns treated as categorical;
- category ordering or the `stringOrderType` setting;
- whether indexers were fitted before or after the split;
- how null or empty strings were handled;
- how categories unseen during fitting were handled; or
- whether ordinal category numbers were passed directly to PCA, LDA, and SVM.

The paper-compatible implementation uses Spark's default frequency ordering, keeps an
extra invalid category, and treats the indices as numeric features. Different choices
change feature geometry and can materially alter PCA, LDA, and SVM results.

### 18. Non-string feature conversion is underspecified

The paper says nearly all source features were retained but does not explain how
timestamps, Booleans, IP addresses, `community_id`, empty strings, or other nonnumeric
values were represented. The implementation converts timestamps and Booleans to
doubles. Its legacy policy indexes all strings; the corrected policy drops identifier-
like fields and one-hot encodes the remaining categorical fields.

### 19. Imputation scope and eligible columns are underspecified

The paper says numeric nulls were replaced with feature means but does not state which
columns contained nulls, whether means were computed before or after splitting, or how
non-numeric missing values were treated. These choices affect both leakage and the
resulting feature matrix.

### 20. Label mappings and positive-class definitions are not published

The paper does not give the numerical mapping for benign and attack labels, the order
of multinomial classes, or the positive class used for binary FPR and AUROC. This
repository maps benign to `0`, attack classes to `1...n`, and treats label `1` as the
binary attack class.

### 21. Metric averaging conventions are not stated

Although the paper supplies elementary formulas, it does not say whether precision,
recall, F1, and FPR are positive-class, macro, micro, or support-weighted statistics.
It also does not define multiclass AUROC aggregation or how One-vs-Rest scores were
combined. These distinctions are material under the severe class imbalance present in
the datasets.

For comparison, this repository interprets the published precision, recall, and F1 as
support-weighted, FPR as attack-class FPR, and multiclass AUROC as support-weighted
one-vs-rest AUROC. It also records macro and per-class measures so this interpretation
can be audited.

### 22. LDA solver details are incomplete

The paper names scikit-learn `LinearDiscriminantAnalysis` but does not report its solver,
version, tolerance, shrinkage, priors, rank handling, or other effective defaults. It
also does not fully specify how transformed features and explained-variance ratios were
obtained. The development reproduction uses scikit-learn's SVD solver. Full-data runs
use a separate sufficient-statistics implementation and are explicitly not exact
scikit-learn reproductions.

### 23. The MapReduce LDA algorithm is not reproducible from the description

The paper mentions map operations for class means and scatter matrices, eigenvalue
calculations, broadcast variables, and 33.2 MiB blocks. It does not provide source code,
complete aggregation formulas, partitioning, numerical solver and regularization
behavior, failure/retry behavior, or enough execution detail to recreate the reported
MapReduce backend.

### 24. The GPU execution path is not established

The paper names the RAPIDS Accelerator for Apache Spark but does not state which
preprocessing, PCA, SVM, One-vs-Rest, or LDA operations actually ran on a GPU. It gives
no RAPIDS configuration, compatibility report, GPU utilization evidence, or explanation
of fallback-to-CPU operations. Therefore the meaning of the reported "GPU training"
and "GPU testing" columns cannot be independently reconstructed.

### 25. CPU and GPU hardware comparability is ambiguous

The paper describes six physical servers, a smaller logical Hadoop/Spark cluster,
specific Spark executor allocations, and separate VMs with four CPUs and 8 GB RAM. It
does not clearly map each reported CPU and GPU timing to those environments or show
that CPU and GPU trials otherwise used equivalent resources. Hardware, memory, and
topology may therefore be confounded with accelerator type.

### 26. Timing boundaries are not defined precisely

"Preprocessing," "training," and "testing" are described conceptually, but the paper
does not state whether timings include Spark lazy evaluation, actions that materialize
results, caching, data loading, host/device transfer, estimator construction, metric
calculation, synchronization, or MapReduce setup. Published timing columns therefore
cannot be mapped reliably to a new implementation's timers.

This repository separately records preprocessing, dimensionality reduction, fitting,
prediction/materialization, and metric calculation.

### 27. Repetition, warm-up, and uncertainty procedures are absent

The paper reports single timing values without saying how many runs were performed,
whether JVM/GPU warm-up occurred, or whether the values are a mean, median, minimum, or
single observation. No variance or confidence interval is provided. The repository's
warm-up and repeated-fit protocol is a local benchmarking improvement, not a recovered
paper procedure.

### 28. Software versions are incomplete for exact reproduction

Spark 3.5.0 and Hadoop 3.3.1 are given, but versions of Python, PySpark, scikit-learn,
Java, CUDA, RAPIDS, GPU drivers, and important numerical libraries are not. Defaults
and numerical behavior can differ across those versions.

### 29. Spark execution details remain incomplete

The paper reports several Spark resource settings but omits other settings that can
affect execution and timing, including deployment mode, dynamic-allocation behavior,
serialization, caching/storage levels, input partitioning, adaptive query execution,
and the exact relationship between the physical and logical clusters.

## Consequence for interpretation

The published tables can be used as reference values, but exact equality is not a
reasonable reproducibility criterion. The most consequential unresolved choices affect
the sampled records, scaling of the baseline, preprocessing leakage, categorical
encoding, metric definitions, hardware allocation, and timing boundaries. Generated
comparison reports therefore retain exclusions and limitations rather than silently
choosing whichever interpretation best matches a published value.
