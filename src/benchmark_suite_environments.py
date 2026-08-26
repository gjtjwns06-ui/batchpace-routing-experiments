"""Environment factories for the BatchPace synthetic + LLM benchmark suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


HORIZON = 24
BATCH_SIZE = 20
SHIFT_BATCH = 12


@dataclass(frozen=True)
class SyntheticSpec:
    environment_id: str
    family: str
    shift: str
    regime: str


FAMILIES = ("sparse", "moderate", "bimodal", "heavy_tail", "rare_jackpot")
SHIFTS = ("mean_up", "mean_down", "variance_up", "tail_reversal")
REGIMES = ("iid", "batch_bursty")


def synthetic_specs() -> list[SyntheticSpec]:
    result = []
    for family in FAMILIES:
        for shift in SHIFTS:
            for regime in REGIMES:
                result.append(SyntheticSpec(
                    f"syn_{len(result) + 1:02d}_{family}_{shift}_{regime}",
                    family, shift, regime,
                ))
    return result


def _base_gain(rng: np.random.Generator, family: str, shape: tuple[int, ...]) -> np.ndarray:
    if family == "sparse":
        active = rng.random(shape) < 0.25
        return active * rng.beta(2.0, 5.0, shape)
    if family == "moderate":
        return rng.beta(2.5, 4.0, shape)
    if family == "bimodal":
        high = rng.random(shape) < 0.35
        return np.where(high, rng.beta(6.0, 2.0, shape), rng.beta(1.5, 7.0, shape))
    if family == "heavy_tail":
        return rng.beta(0.75, 3.5, shape)
    if family == "rare_jackpot":
        jackpot = rng.random(shape) < 0.08
        return np.where(jackpot, rng.beta(12.0, 1.5, shape), rng.beta(0.5, 12.0, shape))
    raise ValueError(f"unknown family {family}")


def _post_shift(rng: np.random.Generator, x: np.ndarray, shift: str) -> np.ndarray:
    if shift == "mean_up":
        return np.clip(0.18 + 1.05 * x, 0, 1)
    if shift == "mean_down":
        return np.clip(0.72 * x - 0.03, 0, 1)
    if shift == "variance_up":
        return np.clip(0.5 + 1.55 * (x - 0.5), 0, 1)
    if shift == "tail_reversal":
        rare = rng.random(x.shape) < 0.12
        return np.where(rare, np.clip(0.72 + 0.28 * rng.random(x.shape), 0, 1),
                        np.clip(0.45 * x, 0, 1))
    raise ValueError(f"unknown shift {shift}")


def generate_synthetic_paths(spec: SyntheticSpec, seed: int, paths: int = 1) -> dict[str, np.ndarray]:
    """Generate true gains, noisy observable scores, uncertainty, and contexts."""
    rng = np.random.default_rng(seed)
    shape = (paths, HORIZON, BATCH_SIZE)
    gain = _base_gain(rng, spec.family, shape)
    gain[:, SHIFT_BATCH:] = _post_shift(rng, gain[:, SHIFT_BATCH:], spec.shift)
    if spec.regime == "batch_bursty":
        latent = rng.normal(0, 0.13, (paths, HORIZON, 1))
        gain = np.clip(gain + latent, 0, 1)
    else:
        latent = np.zeros((paths, HORIZON, 1))
    uncertainty = 0.08 + 0.10 * (1.0 - np.abs(2.0 * gain - 1.0))
    score = np.clip(gain + rng.normal(0, uncertainty), 0, 1)
    phase = np.broadcast_to(
        (np.arange(HORIZON) >= SHIFT_BATCH)[None, :, None], shape
    ).astype(np.float64)
    return {
        "gain": gain,
        "score": score,
        "uncertainty": uncertainty,
        "phase": phase,
        "latent": np.broadcast_to(latent, shape),
    }


def synthetic_manifest() -> list[dict[str, str]]:
    return [asdict(spec) for spec in synthetic_specs()]

