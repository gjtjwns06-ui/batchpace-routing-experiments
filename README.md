# BatchPace Routing Experiments

Reproducible synthetic and local-LLM experiments for comparing global-budget
query-routing policies under batched distribution shift.

## What is included

- 40 synthetic gain-distribution environments
- 40 Qwen/MMLU-Pro-derived environments
- 24 batches per environment, with a shift after batch 12
- 8 global-budget settings and 10 random seeds
- 7 routing policies:
  - BatchPace
  - Oracle BatchPace
  - Sequential Pacing
  - Fixed Threshold
  - Equal Split
  - WISERouter-Offline
  - Robust Batch-Level Query Routing
- Per-environment PNG/SVG figures and an 80-page PDF atlas
- Cached option-label conditional log-likelihoods for Qwen3-0.6B and Qwen3-4B

## Main outputs

- `output/benchmark_suite/`: 80 per-environment PNG/SVG figures
- `output/pdf/batchpace_80_environment_atlas.pdf`: combined atlas
- `results/benchmark_suite/all_summary.csv`: mean and 95% CI by environment,
  budget, and policy
- `results/benchmark_suite/all_metrics.parquet`: all 44,800 policy runs

## Reproduction

See [README_BENCHMARK_SUITE.md](README_BENCHMARK_SUITE.md) for the complete
experimental design, baseline mapping, local llama.cpp commands, and output
layout.

Install the Python dependencies with:

```bash
python -m pip install -r requirements-benchmark.txt
```

The GGUF model weights and llama.cpp binaries are intentionally excluded from
Git. Download the Qwen GGUF files separately and place them under
`data/real_llm/models/` before rerunning local inference.

