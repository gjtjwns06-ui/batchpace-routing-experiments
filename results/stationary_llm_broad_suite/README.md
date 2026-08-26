# Broad-category stationary LLM-derived suite

This replacement experiment uses the 14 broad MMLU-Pro categories rather than
40 fine-grained `src` subjects.

- Categories: biology, business, chemistry, computer science, economics,
  engineering, health, history, law, math, other, philosophy, physics, and
  psychology
- One fixed broad category per environment for all 24 batches
- 20 IID bootstrap query arrivals per batch (480 arrivals per replay path)
- No category transition, no batch-12 change point, and no batch-bursty regime
- 10 replay seeds and budgets `0, 24, 48, 72, 96, 120, 144, 192`
- Cached option scoring: `Qwen3-0.6B-Q8_0` versus `Qwen3-4B-Q4_K_M`

`metrics.parquet` retains all seven implemented policies.  The top-8 figure
uses the five-policy LLM presentation requested by the user.  Ranking is based
on BatchPace's mean gain across nonzero budgets minus the strongest competing
policy's mean gain across those budgets.

The exact unique category-pool sizes and gain composition are recorded in
`environment_manifest.json` and `query_pool_audit.csv`.

