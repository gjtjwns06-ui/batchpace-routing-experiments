"""Policies and dynamic programs for the BatchPace synthetic study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyOutcome:
    true_gain: float
    budget_used: int
    selected: np.ndarray
    batch_allocations: tuple[int, ...]


def _outcome(
    true_gain: np.ndarray,
    selected: np.ndarray,
    allocations: list[int],
) -> PolicyOutcome:
    selected = np.asarray(selected, dtype=np.int64)
    return PolicyOutcome(
        true_gain=float(true_gain[selected].sum()) if selected.size else 0.0,
        budget_used=int(selected.size),
        selected=selected,
        batch_allocations=tuple(int(value) for value in allocations),
    )


def offline_oracle(true_gain: np.ndarray, budget: int) -> PolicyOutcome:
    k = min(int(budget), int(true_gain.size))
    selected = np.argsort(-true_gain, kind="stable")[:k]
    return _outcome(true_gain, selected, [])


def build_batch_value_table(
    calibration_scores: np.ndarray,
    horizon: int,
    budget: int,
) -> np.ndarray:
    """Empirical Bellman table V[remaining_batches, remaining_budget]."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("calibration_scores must have shape [batches, batch_size]")
    if np.any(scores < 0):
        raise ValueError("scores must be non-negative")

    batch_size = scores.shape[1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumulative = np.concatenate(
        [np.zeros((scores.shape[0], 1)), np.cumsum(ordered, axis=1)], axis=1
    )
    value = np.zeros((horizon + 1, budget + 1), dtype=np.float64)

    for remaining_batches in range(1, horizon + 1):
        previous = value[remaining_batches - 1]
        for remaining_budget in range(1, budget + 1):
            max_slots = min(batch_size, remaining_budget)
            slots = np.arange(max_slots + 1)
            candidates = (
                cumulative[:, : max_slots + 1]
                + previous[remaining_budget - slots][None, :]
            )
            value[remaining_batches, remaining_budget] = np.max(
                candidates, axis=1
            ).mean()

    _validate_value_table(value)
    return value


def build_sequential_value_table(
    calibration_scores: np.ndarray,
    horizon: int,
    budget: int,
) -> np.ndarray:
    """Online single-query value table using the empirical score distribution."""
    samples = np.sort(np.asarray(calibration_scores, dtype=np.float64).reshape(-1))
    if samples.size == 0 or np.any(samples < 0):
        raise ValueError("calibration_scores must contain non-negative values")
    prefix = np.concatenate([[0.0], np.cumsum(samples)])
    total = float(prefix[-1])
    count = int(samples.size)
    value = np.zeros((horizon + 1, budget + 1), dtype=np.float64)

    for remaining_queries in range(1, horizon + 1):
        previous = value[remaining_queries - 1]
        for remaining_budget in range(1, budget + 1):
            skip_value = previous[remaining_budget]
            take_future = previous[remaining_budget - 1]
            threshold = skip_value - take_future
            first = int(np.searchsorted(samples, threshold, side="right"))
            tail_count = count - first
            positive_excess = (
                total - float(prefix[first]) - tail_count * threshold
            ) / count
            value[remaining_queries, remaining_budget] = (
                skip_value + max(0.0, positive_excess)
            )

    _validate_value_table(value)
    return value


def _validate_value_table(value: np.ndarray) -> None:
    tolerance = 1e-10
    if not np.allclose(value[0], 0.0):
        raise AssertionError("V[0, b] must be zero")
    if np.min(np.diff(value, axis=0)) < -tolerance:
        raise AssertionError("Value table must be monotone in horizon")
    if np.min(np.diff(value, axis=1)) < -tolerance:
        raise AssertionError("Value table must be monotone in budget")


def run_batchpace(
    scores: np.ndarray,
    true_gain: np.ndarray,
    batch_size: int,
    budget: int,
    value_table: np.ndarray,
) -> PolicyOutcome:
    n_queries = int(scores.size)
    if n_queries % batch_size:
        raise ValueError("Number of queries must be divisible by batch_size")
    horizon = n_queries // batch_size
    if value_table.shape[0] < horizon + 1 or value_table.shape[1] < budget + 1:
        raise ValueError("Value table is too small")

    remaining_budget = int(budget)
    selected: list[int] = []
    allocations: list[int] = []

    for batch_id in range(horizon):
        start = batch_id * batch_size
        stop = start + batch_size
        order = np.argsort(-scores[start:stop], kind="stable")
        ordered_scores = scores[start:stop][order]
        max_slots = min(batch_size, remaining_budget)
        cumulative = np.concatenate([[0.0], np.cumsum(ordered_scores[:max_slots])])
        slots = np.arange(max_slots + 1)
        future_batches = horizon - batch_id - 1
        candidates = (
            cumulative
            + value_table[future_batches, remaining_budget - slots]
        )
        best = float(np.max(candidates))
        # Prefer using more slots when predicted values tie (usually due to clipping at 0).
        allocation = int(
            np.flatnonzero(np.isclose(candidates, best, atol=1e-12, rtol=0.0))[-1]
        )
        chosen = start + order[:allocation]
        selected.extend(int(index) for index in chosen)
        allocations.append(allocation)
        remaining_budget -= allocation

    return _outcome(true_gain, np.asarray(selected), allocations)


def run_sequential(
    scores: np.ndarray,
    true_gain: np.ndarray,
    budget: int,
    value_table: np.ndarray,
    batch_size_for_trace: int,
) -> PolicyOutcome:
    n_queries = int(scores.size)
    remaining_budget = int(budget)
    selected: list[int] = []
    allocations = [0] * (n_queries // batch_size_for_trace)

    for query_id, score in enumerate(scores):
        if remaining_budget <= 0:
            break
        remaining_after = n_queries - query_id - 1
        skip_value = value_table[remaining_after, remaining_budget]
        take_value = float(score) + value_table[remaining_after, remaining_budget - 1]
        if take_value >= skip_value - 1e-12:
            selected.append(query_id)
            allocations[query_id // batch_size_for_trace] += 1
            remaining_budget -= 1

    return _outcome(true_gain, np.asarray(selected), allocations)


def run_fixed_threshold(
    scores: np.ndarray,
    true_gain: np.ndarray,
    batch_size: int,
    budget: int,
    threshold: float,
) -> PolicyOutcome:
    n_queries = int(scores.size)
    horizon = n_queries // batch_size
    remaining_budget = int(budget)
    selected: list[int] = []
    allocations: list[int] = []

    for batch_id in range(horizon):
        start = batch_id * batch_size
        stop = start + batch_size
        local = np.flatnonzero(scores[start:stop] >= threshold)
        if local.size > remaining_budget:
            local_scores = scores[start:stop][local]
            local = local[np.argsort(-local_scores, kind="stable")[:remaining_budget]]
        selected.extend(int(start + index) for index in local)
        allocation = int(local.size)
        allocations.append(allocation)
        remaining_budget -= allocation
        if remaining_budget == 0:
            allocations.extend([0] * (horizon - batch_id - 1))
            break

    return _outcome(true_gain, np.asarray(selected), allocations)


def run_equal_split(
    scores: np.ndarray,
    true_gain: np.ndarray,
    batch_size: int,
    budget: int,
) -> PolicyOutcome:
    n_queries = int(scores.size)
    horizon = n_queries // batch_size
    used = 0
    selected: list[int] = []
    allocations: list[int] = []

    for batch_id in range(horizon):
        target_cumulative = int(np.floor(budget * (batch_id + 1) / horizon + 1e-12))
        allocation = min(target_cumulative - used, batch_size, budget - used)
        start = batch_id * batch_size
        stop = start + batch_size
        order = np.argsort(-scores[start:stop], kind="stable")
        selected.extend(int(start + index) for index in order[:allocation])
        allocations.append(int(allocation))
        used += int(allocation)

    return _outcome(true_gain, np.asarray(selected), allocations)
