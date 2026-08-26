"""Routing policies used by the cached real-LLM replay experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyOutcome:
    true_gain: float
    budget_used: int
    selected: np.ndarray
    batch_allocations: tuple[int, ...]


def make_outcome(
    true_gain: np.ndarray, selected: list[int] | np.ndarray, allocations: list[int]
) -> PolicyOutcome:
    indices = np.asarray(selected, dtype=np.int64)
    return PolicyOutcome(
        true_gain=float(true_gain[indices].sum()) if indices.size else 0.0,
        budget_used=int(indices.size),
        selected=indices,
        batch_allocations=tuple(int(value) for value in allocations),
    )


def offline_oracle(true_gain: np.ndarray, budget: int, batch_size: int) -> PolicyOutcome:
    positive = np.flatnonzero(true_gain > 0)
    order = positive[np.argsort(-true_gain[positive], kind="stable")]
    selected = order[:budget]
    allocations = [0] * (len(true_gain) // batch_size)
    for index in selected:
        allocations[int(index) // batch_size] += 1
    return make_outcome(true_gain, selected, allocations)


def robust_batch_level(
    robust_incremental_value: np.ndarray,
    true_gain: np.ndarray,
    batch_size: int,
    budget: int,
) -> PolicyOutcome:
    """Two-model reduction of the paper's per-batch robust ILP.

    Cheap-model incremental cost is zero and strong-model incremental cost is one.
    The global budget fraction is converted to the original paper's fixed per-batch
    cost cap; for the 5/10/20% settings and batch size 20 this is exactly 1/2/4.
    """
    horizon = len(true_gain) // batch_size
    if budget % horizon:
        raise ValueError("Robust baseline requires an integral per-batch cap")
    cap = budget // horizon
    selected: list[int] = []
    allocations: list[int] = []
    for batch_id in range(horizon):
        start = batch_id * batch_size
        stop = start + batch_size
        local_value = robust_incremental_value[start:stop]
        candidates = np.flatnonzero(local_value > 0)
        order = candidates[np.argsort(-local_value[candidates], kind="stable")]
        chosen = order[:cap]
        selected.extend(int(start + index) for index in chosen)
        allocations.append(int(len(chosen)))
    return make_outcome(true_gain, selected, allocations)


def wiserouter_offline(
    contexts: np.ndarray,
    context_probabilities: np.ndarray,
    cheap_rewards: np.ndarray,
    strong_rewards: np.ndarray,
    true_gain: np.ndarray,
    budget: int,
    batch_size: int,
    seed: int,
) -> PolicyOutcome:
    """WR-Offline adaptive LP specialized to two fixed-cost model actions."""
    rng = np.random.default_rng(seed)
    remaining_budget = int(budget)
    selected: list[int] = []
    allocations = [0] * (len(contexts) // batch_size)
    delta = np.asarray(strong_rewards) - np.asarray(cheap_rewards)

    for index, context in enumerate(contexts):
        if remaining_budget <= 0:
            break
        remaining_rounds = len(contexts) - index
        rho = min(1.0, remaining_budget / remaining_rounds)
        strong_probability = _alp_strong_probability(
            int(context), context_probabilities, delta, rho
        )
        if rng.random() < strong_probability:
            selected.append(index)
            allocations[index // batch_size] += 1
            remaining_budget -= 1

    return make_outcome(true_gain, selected, allocations)


def _alp_strong_probability(
    current_context: int,
    context_probabilities: np.ndarray,
    reward_delta: np.ndarray,
    rho: float,
) -> float:
    probabilities = np.zeros_like(reward_delta, dtype=np.float64)
    remaining = float(rho)
    for context in np.argsort(-reward_delta, kind="stable"):
        if reward_delta[context] <= 0 or remaining <= 0:
            break
        mass = float(context_probabilities[context])
        if mass <= 0:
            continue
        probabilities[context] = min(1.0, remaining / mass)
        remaining -= probabilities[context] * mass
    return float(probabilities[current_context])
