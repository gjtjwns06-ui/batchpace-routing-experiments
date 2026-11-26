# BatchPace 40 + 40 benchmark suite

> This guide covers the original distribution-shift suite. For all five suites,
> including the stationary Beta and LLM experiments, see the
> [project README](README.md) and the Korean [experiment record](docs/EXPERIMENTS.md).

This directory contains a reproducible comparison of seven routing policies on
40 synthetic and 40 Qwen/MMLU-Pro-derived environments. Every environment has
24 batches of 20 queries (480 total), and the workload changes after batch 12.

## Experiment definition

- Budgets: `0, 24, 48, 72, 96, 120, 144, 192` strong-model calls.
- Repetitions: 10 independently sampled test streams per environment.
- Reported uncertainty: mean plus/minus normal-approximation 95% CI over seeds.
- Reward: query-level incremental reward `strong_correct - cheap_correct` for
  LLM experiments, and the generated `[0,1]` gain for synthetic experiments.
- A policy may use less than `B`, but it may never exceed `B`.

The 40 synthetic environments are the Cartesian product of five gain families
(`sparse`, `moderate`, `bimodal`, `heavy_tail`, `rare_jackpot`), four shifts
(`mean_up`, `mean_down`, `variance_up`, `tail_reversal`), and two batch regimes
(`iid`, `batch_bursty`). The exact manifest is written to
`results/benchmark_suite/synthetic_environment_manifest.json`.

The 40 LLM environments use 20 directed MMLU-Pro category transitions and two
batch regimes. Ten category pairs are selected using only calibration-split
mean-gain differences, then evaluated in both directions. Each environment
draws from the source category for batches 1--12 and the target category for
batches 13--24. The bursty regime samples an entire batch from one predicted
difficulty quartile. The exact manifest is written to
`results/benchmark_suite/llm_environment_manifest.json`.

## Option scoring

The cheap and strong models are Qwen3-0.6B-Q8_0 and Qwen3-4B-Q4_K_M. For every
question, llama.cpp receives the same five-shot, category-matched prompt. A
grammar limits the next token to the valid option labels and the truncation
samplers are disabled. The server's post-sampling distribution yields
`log p(A|prompt), ..., log p(J|prompt)` directly. This is constrained candidate
scoring; no generated answer text is parsed to obtain the prediction.

## Policy mapping

- **BatchPace:** predicted incremental gain and a position-aware batch Bellman
  table estimated from calibration paths.
- **Oracle BatchPace:** the same online batch Bellman rule, but current-query
  scores and calibration values are true gains. It is not the hindsight top-B
  oracle.
- **Sequential Pacing:** one-query-at-a-time Bellman pacing.
- **Fixed Threshold:** one calibration quantile per global budget, with a hard
  global budget guard.
- **Equal Split:** cumulative budget is divided evenly across 24 batches; each
  batch sends its top-scored queries.
- **WISERouter-Offline:** the offline adaptive LP specialized to two actions
  with costs 0 and 1. Sixteen K-means contexts estimate context probabilities
  and cheap/strong rewards from the training split.
- **Robust Batch-Level Query Routing:** the original robust batch ILP reduced to
  two models with unit incremental cost and no capacity constraint. It selects
  the largest positive 10th-percentile lower bounds from 100 bootstrap refits
  under an equal per-batch cap. In the synthetic experiment the known 10th-percentile
  Gaussian lower bound is used.

## Reproduction

Start `llama-server` for each model, then run:

```powershell
python scripts/score_options_llamacpp.py --input data/real_llm/splits/mmlu_pro_main_1500.parquet --demonstrations data/real_llm/mmlu_pro/data/validation-00000-of-00001.parquet --output results/benchmark_suite/option_scores_qwen3_0.6b.jsonl --model-id Qwen3-0.6B-Q8_0 --shots 5
python scripts/score_options_llamacpp.py --input data/real_llm/splits/mmlu_pro_main_1500.parquet --demonstrations data/real_llm/mmlu_pro/data/validation-00000-of-00001.parquet --output results/benchmark_suite/option_scores_qwen3_4b.jsonl --model-id Qwen3-4B-Q4_K_M --shots 5
python scripts/run_benchmark_suite.py --part all --seeds 10 --calibration-paths 200
python scripts/plot_benchmark_suite.py
python -m unittest tests.test_benchmark_suite -v
```

Outputs are under `results/benchmark_suite` and `output/benchmark_suite`. There
is one PNG and one SVG per environment; the combined 80-page PDF atlas is under
`output/pdf`.
