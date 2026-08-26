"""Run the 14 broad-category stationary LLM-derived environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_benchmark_suite import (  # noqa: E402
    DEFAULT_BUDGETS,
    evaluate_methods,
    load_llm_frame,
    summarize,
)
from scripts.run_stationary_llm_suite import policy_inputs  # noqa: E402
from src.benchmark_suite_environments import BATCH_SIZE, HORIZON  # noqa: E402
from src.benchmark_suite_policies import (  # noqa: E402
    build_batch_dp,
    build_sequential_dp,
    threshold_from_calibration,
)
from src.stationary_llm_broad_environments import (  # noqa: E402
    broad_stationary_llm_specs,
    broad_stationary_manifest,
    sample_broad_stationary_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--calibration-paths", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--cheap-scores",
        type=Path,
        default=ROOT / "results/benchmark_suite/option_scores_qwen3_0.6b.jsonl",
    )
    parser.add_argument(
        "--strong-scores",
        type=Path,
        default=ROOT / "results/benchmark_suite/option_scores_qwen3_4b.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stationary_llm_broad_suite",
    )
    parser.add_argument("--robust-bootstraps", type=int, default=100)
    return parser.parse_args()


def load_or_build_policy_inputs(
    frame: pd.DataFrame, output: Path, robust_bootstraps: int
) -> dict[str, np.ndarray]:
    question_ids = frame["question_id"].astype(str).to_numpy(dtype=str)
    candidates = (
        output / "policy_inputs.npz",
        ROOT / "results/stationary_llm_suite/policy_inputs.npz",
    )
    for cache_path in candidates:
        if not cache_path.exists():
            continue
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_ids = cached["question_ids"].astype(str)
        except ValueError:
            # Trusted local cache produced by an earlier version that stored
            # question IDs with object dtype; it is immediately re-saved below.
            cached = np.load(cache_path, allow_pickle=True)
            cached_ids = cached["question_ids"].astype(str)
        if np.array_equal(question_ids, cached_ids):
            values = {key: cached[key] for key in cached.files if key != "question_ids"}
            np.savez_compressed(candidates[0], question_ids=question_ids, **values)
            return values
    return policy_inputs(frame, output, robust_bootstraps)


def write_pool_audit(frame: pd.DataFrame, specs: list, output: Path) -> None:
    gains = frame.groupby(["category", "split"], sort=True)["gain"].agg(
        unique_queries="size",
        mean_gain="mean",
        positive_rate=lambda values: (values > 0).mean(),
        negative_rate=lambda values: (values < 0).mean(),
    ).reset_index()
    environment = pd.DataFrame(broad_stationary_manifest(specs))[
        ["environment_id", "category", "category_label"]
    ]
    environment.merge(gains, on="category", validate="one_to_many").to_csv(
        output / "query_pool_audit.csv", index=False
    )


def main() -> None:
    args = parse_args()
    if args.seeds < 2:
        raise ValueError("at least two seeds are required for confidence intervals")
    if any(b < 0 or b > HORIZON * BATCH_SIZE for b in args.budgets):
        raise ValueError(f"budgets must be between 0 and {HORIZON * BATCH_SIZE}")

    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_llm_frame(args.cheap_scores, args.strong_scores)
    specs = broad_stationary_llm_specs(frame)
    (args.output / "environment_manifest.json").write_text(
        json.dumps(broad_stationary_manifest(specs), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_pool_audit(frame, specs, args.output)
    inputs = load_or_build_policy_inputs(frame, args.output, args.robust_bootstraps)
    gain_all = frame["gain"].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []

    for position, spec in enumerate(specs, 1):
        print(f"[broad stationary llm {position:02d}/{len(specs)}] {spec.category_label}", flush=True)
        calibration_idx = sample_broad_stationary_paths(
            frame,
            spec,
            "calibration",
            seed=10_000_000 + position,
            paths=args.calibration_paths,
        )
        calibration_score = inputs["score"][calibration_idx]
        calibration_gain = gain_all[calibration_idx]
        calibration_artifacts = {
            "score_batch_dp": build_batch_dp(calibration_score, max(args.budgets)),
            "gain_batch_dp": build_batch_dp(calibration_gain, max(args.budgets)),
            "sequential_dp": build_sequential_dp(calibration_score, max(args.budgets)),
            "thresholds": {
                budget: threshold_from_calibration(calibration_score, budget)
                for budget in args.budgets
            },
        }

        for seed in range(args.seeds):
            idx = sample_broad_stationary_paths(
                frame,
                spec,
                "test",
                seed=11_000_000 + position * 1000 + seed,
                paths=1,
            ).reshape(-1)
            evaluate_methods(
                gain=gain_all[idx],
                score=inputs["score"][idx],
                robust_score=inputs["robust"][idx],
                contexts=inputs["contexts"][idx],
                pi=inputs["pi"],
                cheap_reward=inputs["cheap_reward"],
                strong_reward=inputs["strong_reward"],
                calibration_score=calibration_score,
                calibration_gain=calibration_gain,
                budgets=args.budgets,
                seed=seed,
                environment_type="stationary_llm_broad",
                environment_id=spec.environment_id,
                rows=rows,
                calibration_artifacts=calibration_artifacts,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_parquet(args.output / "metrics.parquet", index=False)
    metrics.drop(columns="batch_allocations").to_csv(args.output / "metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output / "summary.csv", index=False)
    summary.to_json(
        args.output / "summary.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )
    print(
        f"completed {len(specs)} broad categories, {args.seeds} seeds, "
        f"{len(args.budgets)} budgets, {metrics['method'].nunique()} methods",
        flush=True,
    )


if __name__ == "__main__":
    main()
