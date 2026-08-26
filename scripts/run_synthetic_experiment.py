"""Run the full Main Figure 1 synthetic experiment after pilot checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_policies import (  # noqa: E402
    PolicyOutcome,
    build_batch_value_table,
    build_sequential_value_table,
    offline_oracle,
    run_batchpace,
    run_equal_split,
    run_fixed_threshold,
    run_sequential,
)


DATA_DIR = ROOT / "data" / "synthetic_main_figure_1"
RESULT_DIR = ROOT / "results" / "synthetic_main_figure_1"
EPSILONS = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
BATCH_SIZES = (1, 5, 10, 20, 50, 100)
ENVIRONMENTS = ("iid", "bursty")
N_QUERIES = 1_000
BUDGET = 50


class TableCache:
    def __init__(self) -> None:
        self.batch: dict[tuple[Any, ...], np.ndarray] = {}
        self.sequential: dict[tuple[Any, ...], np.ndarray] = {}

    def batch_table(
        self,
        key: tuple[Any, ...],
        calibration_scores: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        if key not in self.batch:
            self.batch[key] = build_batch_value_table(
                calibration_scores, horizon=horizon, budget=BUDGET
            )
        return self.batch[key]

    def sequential_table(
        self,
        key: tuple[Any, ...],
        calibration_scores: np.ndarray,
    ) -> np.ndarray:
        if key not in self.sequential:
            self.sequential[key] = build_sequential_value_table(
                calibration_scores, horizon=N_QUERIES, budget=BUDGET
            )
        return self.sequential[key]


def arrays(data: Any, batch_size: int, environment: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"m{batch_size:03d}_{environment}"
    return (
        data[f"{prefix}_true_gain"],
        data[f"{prefix}_high_state"],
        data[f"{prefix}_unit_noise"],
    )


def run_setting(
    *,
    panel: str,
    test_data: Any,
    calibration_data: Any,
    environment: str,
    batch_size: int,
    epsilon: float,
    episodes: int,
    cache: TableCache,
    keep_trace: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_gain, _, test_noise = arrays(test_data, batch_size, environment)
    calibration_gain, _, calibration_noise = arrays(
        calibration_data, batch_size, environment
    )
    test_score = np.clip(test_gain + epsilon * test_noise, 0.0, 1.0)
    calibration_score = np.clip(
        calibration_gain + epsilon * calibration_noise, 0.0, 1.0
    )
    horizon = N_QUERIES // batch_size

    oracle_v = cache.batch_table(
        ("oracle", environment, batch_size), calibration_gain, horizon
    )
    predicted_v = cache.batch_table(
        ("predicted", environment, batch_size, epsilon),
        calibration_score,
        horizon,
    )
    sequential_v = cache.sequential_table(
        ("sequential", environment, batch_size, epsilon), calibration_score
    )
    threshold = float(
        np.quantile(calibration_score.reshape(-1), 1.0 - BUDGET / N_QUERIES, method="higher")
    )

    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for episode in range(episodes):
        gain = test_gain[episode].astype(np.float64, copy=False)
        score = test_score[episode].astype(np.float64, copy=False)
        offline = offline_oracle(gain, BUDGET)
        outcomes: dict[str, PolicyOutcome] = {
            "Oracle BatchPace": run_batchpace(
                gain, gain, batch_size, BUDGET, oracle_v
            ),
            "Predicted BatchPace": run_batchpace(
                score, gain, batch_size, BUDGET, predicted_v
            ),
            "Sequential": run_sequential(
                score, gain, BUDGET, sequential_v, batch_size
            ),
            "Fixed Threshold": run_fixed_threshold(
                score, gain, batch_size, BUDGET, threshold
            ),
            "Equal Split": run_equal_split(
                score, gain, batch_size, BUDGET
            ),
        }

        for method, outcome in outcomes.items():
            recovery = outcome.true_gain / offline.true_gain
            record = {
                "panel": panel,
                "environment": environment,
                "episode": episode,
                "epsilon": float(epsilon),
                "batch_size": int(batch_size),
                "n_batches": int(horizon),
                "budget": BUDGET,
                "method": method,
                "true_gain": outcome.true_gain,
                "offline_gain": offline.true_gain,
                "oracle_recovery": recovery,
                "budget_used": outcome.budget_used,
                "threshold": threshold if method == "Fixed Threshold" else None,
            }
            validate_record(record)
            records.append(record)

            if keep_trace and episode == 0:
                remaining = BUDGET
                for batch_id, allocated in enumerate(outcome.batch_allocations):
                    traces.append({
                        "panel": panel,
                        "environment": environment,
                        "epsilon": float(epsilon),
                        "batch_size": int(batch_size),
                        "episode": episode,
                        "method": method,
                        "batch_id": batch_id,
                        "remaining_budget_before": remaining,
                        "allocated_slots": allocated,
                        "remaining_budget_after": remaining - allocated,
                    })
                    remaining -= allocated

    return records, traces


def validate_record(record: dict[str, Any]) -> None:
    tolerance = 1e-8
    if record["budget_used"] > record["budget"]:
        raise AssertionError(f"Budget violation: {record}")
    if record["true_gain"] > record["offline_gain"] + tolerance:
        raise AssertionError(f"Policy exceeds offline oracle: {record}")
    if not -tolerance <= record["oracle_recovery"] <= 1.0 + tolerance:
        raise AssertionError(f"Recovery outside [0,1]: {record}")


def run_all(episodes: int, cache: TableCache) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fig1a_test = np.load(DATA_DIR / "fig1a_test.npz")
    fig1a_cal = np.load(DATA_DIR / "fig1a_calibration.npz")
    fig1b_test = np.load(DATA_DIR / "fig1b_test.npz")
    fig1b_cal = np.load(DATA_DIR / "fig1b_calibration.npz")

    for environment in ENVIRONMENTS:
        for suffix in ("true_gain", "high_state", "unit_noise"):
            key = f"m020_{environment}_{suffix}"
            if not np.array_equal(fig1a_test[key], fig1b_test[key]):
                raise AssertionError(f"Panel datasets disagree for {key}")
            if not np.array_equal(fig1a_cal[key], fig1b_cal[key]):
                raise AssertionError(f"Panel calibration datasets disagree for {key}")

    all_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for epsilon in EPSILONS:
            records, traces = run_setting(
                panel="fig1a",
                test_data=fig1a_test,
                calibration_data=fig1a_cal,
                environment=environment,
                batch_size=20,
                epsilon=epsilon,
                episodes=episodes,
                cache=cache,
                keep_trace=(environment == "bursty" and epsilon == 0.10),
            )
            all_records.extend(records)
            all_traces.extend(traces)

    for environment in ENVIRONMENTS:
        for batch_size in BATCH_SIZES:
            records, traces = run_setting(
                panel="fig1b",
                test_data=fig1b_test,
                calibration_data=fig1b_cal,
                environment=environment,
                batch_size=batch_size,
                epsilon=0.10,
                episodes=episodes,
                cache=cache,
                keep_trace=(environment == "bursty" and batch_size == 20),
            )
            all_records.extend(records)
            all_traces.extend(traces)

    return all_records, all_traces


def pilot_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["panel"],
            record["environment"],
            record["epsilon"],
            record["batch_size"],
            record["episode"],
        )
        by_key[key][record["method"]] = record

    epsilon_zero_diffs = []
    m_one_diffs = []
    for key, methods in by_key.items():
        panel, _, epsilon, batch_size, _ = key
        if panel == "fig1a" and epsilon == 0.0:
            epsilon_zero_diffs.append(abs(
                methods["Predicted BatchPace"]["true_gain"]
                - methods["Oracle BatchPace"]["true_gain"]
            ))
        if panel == "fig1b" and batch_size == 1:
            m_one_diffs.append(abs(
                methods["Predicted BatchPace"]["true_gain"]
                - methods["Sequential"]["true_gain"]
            ))

    max_epsilon_zero_diff = max(epsilon_zero_diffs)
    max_m_one_diff = max(m_one_diffs)
    if max_epsilon_zero_diff > 1e-8:
        raise AssertionError("epsilon=0 Predicted and Oracle BatchPace differ")
    if max_m_one_diff > 1e-8:
        raise AssertionError("m=1 BatchPace and Sequential differ")

    return {
        "records_checked": len(records),
        "max_epsilon_zero_gain_difference": max_epsilon_zero_diff,
        "max_m_one_gain_difference": max_m_one_diff,
        "all_budget_constraints_satisfied": True,
        "all_recoveries_in_unit_interval": True,
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["panel"],
            record["environment"],
            record["epsilon"],
            record["batch_size"],
            record["method"],
        )
        groups[key].append(record)

    summary = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        panel, environment, epsilon, batch_size, method = key
        recovery = np.asarray([row["oracle_recovery"] for row in rows])
        gain = np.asarray([row["true_gain"] for row in rows])
        budget_used = np.asarray([row["budget_used"] for row in rows])
        n = int(recovery.size)
        ci95 = 1.96 * float(recovery.std(ddof=1)) / np.sqrt(n)
        summary.append({
            "panel": panel,
            "environment": environment,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "method": method,
            "episodes": n,
            "mean_oracle_recovery": float(recovery.mean()),
            "ci95_oracle_recovery": ci95,
            "mean_true_gain": float(gain.mean()),
            "mean_budget_used": float(budget_used.mean()),
        })
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-episodes", type=int, default=10)
    args = parser.parse_args()

    cache = TableCache()
    print("Building value tables and running pilot checks...", flush=True)
    pilot_records, _ = run_all(args.pilot_episodes, cache)
    checks = pilot_checks(pilot_records)
    print(json.dumps({"pilot": checks}, indent=2), flush=True)

    print("Pilot passed. Running all 250 episodes per setting...", flush=True)
    records, traces = run_all(250, cache)
    checks["full_records_checked"] = len(records)
    summary = aggregate(records)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(RESULT_DIR / "episode_metrics.jsonl", records)
    write_jsonl(RESULT_DIR / "representative_policy_traces.jsonl", traces)
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    focus = [
        row for row in summary
        if row["environment"] == "bursty"
        and (
            (row["panel"] == "fig1a" and row["method"] in {"Predicted BatchPace", "Sequential"})
            or (row["panel"] == "fig1b" and row["batch_size"] in {1, 20, 100} and row["method"] in {"Predicted BatchPace", "Sequential"})
        )
    ]
    print(json.dumps({"outputs": str(RESULT_DIR), "focus_summary": focus}, indent=2), flush=True)


if __name__ == "__main__":
    main()
