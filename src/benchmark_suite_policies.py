"""Policy implementations shared by the 40+40 BatchPace benchmark suite.

All policies see a stream of 24 batches.  Routing a query to the strong model
costs one unit and realizes ``gain``; the cheap action costs zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcome:
    gain: float
    budget_used: int
    selected: np.ndarray
    allocations: tuple[int, ...]


def _outcome(gain: np.ndarray, selected: list[int] | np.ndarray, batch_size: int) -> Outcome:
    idx = np.asarray(selected, dtype=np.int64)
    allocations = np.zeros(len(gain) // batch_size, dtype=np.int64)
    if idx.size:
        np.add.at(allocations, idx // batch_size, 1)
    return Outcome(float(gain[idx].sum()) if idx.size else 0.0, int(idx.size), idx,
                   tuple(int(x) for x in allocations))


def offline_oracle(gain: np.ndarray, budget: int, batch_size: int) -> Outcome:
    """Hindsight upper bound, used only for validation (not a requested curve)."""
    positive = np.flatnonzero(gain > 0)
    order = positive[np.argsort(-gain[positive], kind="stable")]
    return _outcome(gain, order[:budget], batch_size)


def build_batch_dp(calibration: np.ndarray, budget: int) -> np.ndarray:
    """Position-aware batch Bellman table.

    ``calibration`` has shape [sample paths, batches, batch size].  V[t,b] is
    the expected future score from batches t..H-1 with b units remaining.
    """
    x = np.asarray(calibration, dtype=np.float64)
    if x.ndim != 3 or np.any(~np.isfinite(x)):
        raise ValueError("calibration must be finite [paths,batches,batch_size]")
    paths, horizon, batch_size = x.shape
    ordered = np.sort(x, axis=2)[:, :, ::-1]
    cumulative = np.concatenate(
        [np.zeros((paths, horizon, 1)), np.cumsum(ordered, axis=2)], axis=2
    )
    value = np.zeros((horizon + 1, budget + 1), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        for b in range(1, budget + 1):
            k = np.arange(min(batch_size, b) + 1)
            candidates = cumulative[:, t, : len(k)] + value[t + 1, b - k][None, :]
            value[t, b] = np.max(candidates, axis=1).mean()
    return value


def build_sequential_dp(calibration: np.ndarray, budget: int) -> np.ndarray:
    """Position-aware one-query Bellman table for Sequential Pacing."""
    x = np.asarray(calibration, dtype=np.float64)
    if x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    if x.ndim != 2:
        raise ValueError("calibration must have shape [paths,queries]")
    _, horizon = x.shape
    value = np.zeros((horizon + 1, budget + 1), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        take = x[:, t, None] + value[t + 1, :budget][None, :]
        skip = value[t + 1, 1:budget + 1][None, :]
        value[t, 1:budget + 1] = np.maximum(skip, take).mean(axis=0)
    return value


def run_batchpace(scores: np.ndarray, gain: np.ndarray, batch_size: int,
                  budget: int, value: np.ndarray) -> Outcome:
    selected: list[int] = []
    remaining = int(budget)
    horizon = len(gain) // batch_size
    for t in range(horizon):
        start = t * batch_size
        order = np.argsort(-scores[start:start + batch_size], kind="stable")
        local = scores[start:start + batch_size][order]
        k = np.arange(min(batch_size, remaining) + 1)
        cumulative = np.concatenate([[0.0], np.cumsum(local[: len(k) - 1])])
        objective = cumulative + value[t + 1, remaining - k]
        best = np.flatnonzero(np.isclose(objective, objective.max(), atol=1e-12))[-1]
        selected.extend(int(start + i) for i in order[:best])
        remaining -= int(best)
    return _outcome(gain, selected, batch_size)


def run_sequential(scores: np.ndarray, gain: np.ndarray, batch_size: int,
                   budget: int, value: np.ndarray) -> Outcome:
    selected: list[int] = []
    remaining = int(budget)
    for t, score in enumerate(scores):
        if remaining <= 0:
            break
        skip = value[t + 1, remaining]
        take = float(score) + value[t + 1, remaining - 1]
        if take >= skip - 1e-12:
            selected.append(t)
            remaining -= 1
    return _outcome(gain, selected, batch_size)


def run_fixed_threshold(scores: np.ndarray, gain: np.ndarray, batch_size: int,
                        budget: int, threshold: float) -> Outcome:
    """Static threshold with a hard global budget and stable stream order."""
    selected: list[int] = []
    remaining = int(budget)
    for t, score in enumerate(scores):
        if remaining <= 0:
            break
        if score >= threshold:
            selected.append(t)
            remaining -= 1
    return _outcome(gain, selected, batch_size)


def threshold_from_calibration(calibration: np.ndarray, budget: int) -> float:
    fraction = budget / calibration.shape[1] / calibration.shape[2]
    if fraction <= 0:
        return np.inf
    return float(np.quantile(calibration, 1.0 - fraction, method="higher"))


def run_equal_split(scores: np.ndarray, gain: np.ndarray, batch_size: int,
                    budget: int) -> Outcome:
    selected: list[int] = []
    used = 0
    horizon = len(gain) // batch_size
    for t in range(horizon):
        cumulative_target = int(np.floor(budget * (t + 1) / horizon + 1e-12))
        allocation = min(batch_size, cumulative_target - used)
        start = t * batch_size
        order = np.argsort(-scores[start:start + batch_size], kind="stable")
        selected.extend(int(start + i) for i in order[:allocation])
        used += allocation
    return _outcome(gain, selected, batch_size)


def run_robust_batch(robust_scores: np.ndarray, gain: np.ndarray, batch_size: int,
                     budget: int) -> Outcome:
    """Robust Batch-Level Query Routing reduced to two unit-cost actions.

    The original per-batch robust optimization becomes top-k selection by the
    lower confidence bound of incremental reward under an equal per-batch cap.
    """
    selected: list[int] = []
    horizon = len(gain) // batch_size
    used = 0
    for t in range(horizon):
        cumulative_target = int(np.floor(budget * (t + 1) / horizon + 1e-12))
        cap = min(batch_size, cumulative_target - used)
        start = t * batch_size
        local = robust_scores[start:start + batch_size]
        positive = np.flatnonzero(local > 0)
        order = positive[np.argsort(-local[positive], kind="stable")]
        chosen = order[:cap]
        selected.extend(int(start + i) for i in chosen)
        used += len(chosen)
    return _outcome(gain, selected, batch_size)


def _alp_probability(context: int, pi: np.ndarray, delta: np.ndarray, rho: float) -> float:
    probabilities = np.zeros_like(delta, dtype=np.float64)
    remaining = float(rho)
    for j in np.argsort(-delta, kind="stable"):
        if delta[j] <= 0 or remaining <= 0:
            break
        if pi[j] > 0:
            probabilities[j] = min(1.0, remaining / pi[j])
            remaining -= probabilities[j] * pi[j]
    return float(probabilities[context])


def run_wiserouter(contexts: np.ndarray, pi: np.ndarray, cheap_reward: np.ndarray,
                   strong_reward: np.ndarray, gain: np.ndarray, batch_size: int,
                   budget: int, seed: int) -> Outcome:
    """WISERouter's offline adaptive LP, specialized to two models/costs 0 and 1."""
    rng = np.random.default_rng(seed)
    delta = np.asarray(strong_reward) - np.asarray(cheap_reward)
    selected: list[int] = []
    remaining = int(budget)
    for t, context in enumerate(contexts):
        if remaining <= 0:
            break
        rho = min(1.0, remaining / (len(gain) - t))
        if rng.random() < _alp_probability(int(context), pi, delta, rho):
            selected.append(t)
            remaining -= 1
    return _outcome(gain, selected, batch_size)
