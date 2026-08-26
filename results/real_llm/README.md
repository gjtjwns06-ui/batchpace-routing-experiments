# Main Figure 2: real-LLM budget-performance experiment

## Completed protocol

- Dataset: `TIGER-Lab/MMLU-Pro`, fixed 1,500-query stratified universe.
- Models: `Qwen3-0.6B-Q8_0` (cheap) and `Qwen3-4B-Q4_K_M` (strong).
- Prompt: five category-matched MMLU-Pro validation demonstrations, thinking disabled,
  deterministic A-J label output.
- Cached replay: both models were run once on every query. Routing policies never call the
  models again and never receive test labels or strong outputs as inputs.
- Repeated evaluation: five stratified 500/500/500 train/calibration/test partitions. Within
  every partition, random and category-bursty streams each use 20 order seeds. The reported
  mean is the mean of the five split means; the 95% CI is computed across those split means.
- Batch size: 20. Strong-model budgets: 5%, 10%, and 20% (25, 50, and 100 slots).

## Policy implementations

- **BatchPace**: Ridge gain predictor using cheap-model A-J probabilities, entropy, margin,
  question length, option count, and category one-hot features. The empirical future-value
  table is built only from the calibration partition.
- **Sequential**: the same Ridge score and calibration distribution as BatchPace, with
  single-query dynamic pacing. This is the controlled allocator comparison.
- **WISERouter-Offline reproduction**: Qwen3-Embedding-0.6B query embeddings, PCA to 64
  dimensions, `J=16` K-means contexts, five K-means seeds, train-only context/action reward
  means, and the paper's adaptive LP specialized analytically to two fixed-cost actions.
- **Robust Batch-Level reproduction**: separate performance estimators for the two models,
  100 bootstrap refits, 10% lower prediction bounds, and per-batch strong-model caps of
  1/2/4. The estimators use histogram gradient boosting rather than the paper's XGBoost/kNN;
  hardware capacity is non-binding in this two-model replay.
- **Offline Oracle**: selects up to `K` test queries with positive realized gain. It is an
  evaluation-only upper bound.

WISERouter and Robust Batch-Level are paper-based reproductions/adaptations, not executions
of author-released repositories.

## Key result at 10% strong-model budget

Accuracy gain is measured in percentage points over cheap-only inference.

| Policy | Random stream | Category-bursty stream |
|---|---:|---:|
| Offline Oracle | 10.00 | 10.00 |
| BatchPace | 3.18 | 2.92 |
| Sequential | 3.16 | 2.90 |
| WISERouter-Offline | 2.13 | 2.04 |
| Robust Batch-Level | 2.24 | 1.75 |

The fixed 1,500-query universe has 22.93% cheap-model accuracy and 40.67% strong-model
accuracy. Query-level realized gain counts are `+1: 391`, `0: 984`, and `-1: 125`.

## Interpretation and limitations

BatchPace is the strongest deployable policy at 10% in both stream settings, but its margin
over Sequential is only 0.02 percentage points. Therefore this run supports competitiveness
against the two requested external baselines, but does **not** establish a meaningful advantage
over the controlled Sequential allocator.

The main bottleneck is score quality: Ridge test Spearman correlations range from about 0.04
to 0.11 across splits. A train-only comparison of richer predictors did not provide a reliable
improvement, so the preregistered minimal Ridge model was retained.

The inference protocol uses deterministic constrained label generation and extracts label
log-probabilities. It does not compute the full conditional log-likelihood of every candidate
answer sequence, which the experiment guide identifies as the highest-reproducibility option.

## Files

- `policy_metrics.csv`: every split/order/policy replay.
- `policy_summary.csv`: hierarchical mean and 95% CI used in the figure.
- `main_table_10pct.csv`: compact 10% budget results.
- `diagnostics.json`: split-specific predictor and robust-estimator diagnostics.
- `policy_prediction_arrays.npz`: cached policy score arrays for all five splits.
- `gain_predictor_comparison.json`: train-only predictor comparison and post-selection test
  diagnostic.
