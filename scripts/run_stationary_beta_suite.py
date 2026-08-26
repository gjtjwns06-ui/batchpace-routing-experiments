"""Run 40 stationary single-Beta synthetic routing environments."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON  # noqa: E402
from src.benchmark_suite_policies import (  # noqa: E402
    Outcome,
    build_batch_dp,
    offline_oracle,
    run_batchpace,
    run_equal_split,
    run_fixed_threshold,
    run_robust_batch,
    run_wiserouter,
    threshold_from_calibration,
)
from src.stationary_beta_environments import (  # noqa: E402
    StationaryBetaSpec,
    generate_stationary_beta_paths,
    stationary_beta_manifest,
    stationary_beta_specs,
)


DEFAULT_BUDGETS = (0, 24, 48, 72, 96, 120, 144, 192)
METHODS = (
    "BatchPace",
    "Oracle BatchPace",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--calibration-paths", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "stationary_beta_suite",
    )
    return parser.parse_args()


def check_outcome(method: str, outcome: Outcome, hindsight: Outcome, budget: int) -> None:
    if outcome.budget_used > budget:
        raise AssertionError(f"{method} used {outcome.budget_used}>{budget}")
    if outcome.gain > hindsight.gain + 1e-9:
        raise AssertionError(f"{method} exceeds the hindsight upper bound")


def context_model(
    spec: StationaryBetaSpec,
) -> tuple[KMeans, np.ndarray, np.ndarray, np.ndarray]:
    stable = zlib.crc32(spec.environment_id.encode("utf-8"))
    train = generate_stationary_beta_paths(
        spec,
        seed=1_100_000 + stable % 100_000,
        paths=30,
    )
    features = np.column_stack(
        [
            train["score"].ravel(),
            train["uncertainty"].ravel(),
            train["phase"].ravel(),
            train["latent"].ravel(),
        ]
    )
    model = KMeans(n_clusters=16, n_init=10, random_state=0).fit(features)
    contexts = model.labels_
    gain = train["gain"].ravel()
    counts = np.bincount(contexts, minlength=16)
    pi = counts / counts.sum()
    cheap_reward = np.zeros(16, dtype=np.float64)
    strong_reward = np.zeros(16, dtype=np.float64)
    for context in range(16):
        strong_reward[context] = gain[contexts == context].mean()
    return model, pi, cheap_reward, strong_reward


def add_record(
    rows: list[dict[str, object]],
    spec: StationaryBetaSpec,
    seed: int,
    budget: int,
    method: str,
    outcome: Outcome,
) -> None:
    rows.append(
        {
            "environment_type": "stationary_beta",
            "environment_id": spec.environment_id,
            "environment_number": spec.environment_number,
            "alpha": spec.alpha,
            "beta": spec.beta,
            "theoretical_mean": spec.theoretical_mean,
            "theoretical_variance": spec.theoretical_variance,
            "seed": seed,
            "budget": budget,
            "budget_fraction": budget / (HORIZON * BATCH_SIZE),
            "method": method,
            "total_realized_gain": outcome.gain,
            "budget_used": outcome.budget_used,
            "batch_allocations": json.dumps(outcome.allocations),
        }
    )


def run_environment(
    spec: StationaryBetaSpec,
    position: int,
    seeds: int,
    calibration_paths: int,
    budgets: list[int],
    rows: list[dict[str, object]],
) -> None:
    calibration = generate_stationary_beta_paths(
        spec,
        seed=2_100_000 + position,
        paths=calibration_paths,
    )
    max_budget = max(budgets)
    score_batch_dp = build_batch_dp(calibration["score"], max_budget)
    gain_batch_dp = build_batch_dp(calibration["gain"], max_budget)
    thresholds = {
        budget: threshold_from_calibration(calibration["score"], budget)
        for budget in budgets
    }
    model, pi, cheap_reward, strong_reward = context_model(spec)

    for seed in range(seeds):
        data = generate_stationary_beta_paths(
            spec,
            seed=3_100_000 + position * 1000 + seed,
            paths=1,
        )
        gain = data["gain"].reshape(-1)
        score = data["score"].reshape(-1)
        uncertainty = data["uncertainty"].reshape(-1)
        features = np.column_stack(
            [
                score,
                uncertainty,
                data["phase"].reshape(-1),
                data["latent"].reshape(-1),
            ]
        )
        contexts = model.predict(features)
        robust_score = score - 1.2815515655446004 * uncertainty

        for budget in budgets:
            hindsight = offline_oracle(gain, budget, BATCH_SIZE)
            outcomes = {
                "BatchPace": run_batchpace(
                    score, gain, BATCH_SIZE, budget, score_batch_dp
                ),
                "Oracle BatchPace": run_batchpace(
                    gain, gain, BATCH_SIZE, budget, gain_batch_dp
                ),
                "Fixed Threshold": run_fixed_threshold(
                    score, gain, BATCH_SIZE, budget, thresholds[budget]
                ),
                "Equal Split": run_equal_split(score, gain, BATCH_SIZE, budget),
                "WISERouter-Offline": run_wiserouter(
                    contexts,
                    pi,
                    cheap_reward,
                    strong_reward,
                    gain,
                    BATCH_SIZE,
                    budget,
                    seed=(
                        8_500_000
                        + position * 100_000
                        + seed * 1000
                        + budget
                    ),
                ),
                "Robust Batch-Level Query Routing": run_robust_batch(
                    robust_score, gain, BATCH_SIZE, budget
                ),
            }
            for method in METHODS:
                outcome = outcomes[method]
                check_outcome(method, outcome, hindsight, budget)
                add_record(rows, spec, seed, budget, method, outcome)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "environment_id",
        "environment_number",
        "alpha",
        "beta",
        "theoretical_mean",
        "theoretical_variance",
        "budget",
        "budget_fraction",
        "method",
    ]
    grouped = metrics.groupby(group_columns, sort=True)["total_realized_gain"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    return summary


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.calibration_paths <= 0:
        raise ValueError("seeds and calibration-paths must be positive")
    budgets = sorted(set(args.budgets))
    if budgets[0] < 0 or budgets[-1] > HORIZON * BATCH_SIZE:
        raise ValueError("budget must be in [0, 480]")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment_manifest.json").write_text(
        json.dumps(stationary_beta_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    specs = stationary_beta_specs()
    for position, spec in enumerate(specs, 1):
        print(
            f"[stationary Beta {position:02d}/40] "
            f"Beta({spec.alpha:g},{spec.beta:g})",
            flush=True,
        )
        run_environment(
            spec,
            position,
            seeds=args.seeds,
            calibration_paths=args.calibration_paths,
            budgets=budgets,
            rows=rows,
        )

    metrics = pd.DataFrame(rows)
    expected = 40 * args.seeds * len(budgets) * len(METHODS)
    if len(metrics) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(metrics)}")
    metrics.to_parquet(args.output / "metrics.parquet", index=False)
    metrics.drop(columns="batch_allocations").to_csv(
        args.output / "metrics.csv", index=False
    )
    summary = summarize(metrics)
    summary.to_csv(args.output / "summary.csv", index=False)
    summary.to_json(
        args.output / "summary.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )
    print(f"Wrote {len(metrics):,} raw rows and {len(summary):,} summary rows")


if __name__ == "__main__":
    main()
