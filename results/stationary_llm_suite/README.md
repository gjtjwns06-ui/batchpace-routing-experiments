# Stationary LLM-derived suite

This directory contains the replacement LLM-derived experiment with no subject
transition and no batch-bursty regime.

## Environment definition

- 40 fine-grained MMLU-Pro source subjects are selected using question-pool
  availability only (never test gain).
- One environment uses exactly one fixed `src` subject for all 24 batches.
- Every batch contains 20 IID bootstrap draws from the same split-specific
  subject pool, so a replay path contains 480 query arrivals.
- Sampling is with replacement because the cached option-scored subject pools
  are smaller than 480 queries.  The unique pool sizes are reported in
  `environment_manifest.json` and `query_pool_audit.csv`.
- There is no change point at batch 12, no subject arrow, and no batch-bursty
  construction.

## LLM-derived gain

The experiment reuses the existing constrained option-label conditional
log-likelihood caches:

- low model: `Qwen3-0.6B-Q8_0`
- high model: `Qwen3-4B-Q4_K_M`
- query gain: `high_model_correct - low_model_correct`

Each environment is evaluated with 10 replay seeds at global budgets
`0, 24, 48, 72, 96, 120, 144, 192`.  The raw result file retains all seven
implemented policies.  The figures show the five-policy LLM presentation
requested by the user: BatchPace, Fixed Threshold, Equal Split,
WISERouter-Offline, and Robust Batch-Level Query Routing.

## Files

- `environment_manifest.json`: exact subject and pool definition for 40 environments
- `query_pool_audit.csv`: unique query count and gain composition by split
- `metrics.parquet` / `metrics.csv`: seed-level policy results
- `summary.csv` / `summary.json`: mean, standard deviation, and 95% CI
- `policy_inputs.npz`: cached policy scores used for reproducible reruns

Run again with:

```powershell
python scripts/run_stationary_llm_suite.py
python scripts/plot_stationary_llm_suite.py
```

