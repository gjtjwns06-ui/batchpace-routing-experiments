"""Forty stationary single-Beta gain environments.

Every environment contains 24 IID batches drawn from one unchanged
``Beta(alpha, beta)`` distribution.  There is no phase shift, mixture, point
mass, or batch-level latent effect in this suite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON


ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
BETAS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


@dataclass(frozen=True)
class StationaryBetaSpec:
    environment_id: str
    environment_number: int
    alpha: float
    beta: float
    theoretical_mean: float
    theoretical_variance: float


def _parameter_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def stationary_beta_specs() -> list[StationaryBetaSpec]:
    result = []
    for alpha in ALPHAS:
        for beta in BETAS:
            number = len(result) + 1
            total = alpha + beta
            result.append(
                StationaryBetaSpec(
                    environment_id=(
                        f"beta_{number:02d}_a{_parameter_tag(alpha)}_"
                        f"b{_parameter_tag(beta)}"
                    ),
                    environment_number=number,
                    alpha=alpha,
                    beta=beta,
                    theoretical_mean=alpha / total,
                    theoretical_variance=(alpha * beta) / (total**2 * (total + 1.0)),
                )
            )
    if len(result) != 40:
        raise AssertionError(f"Expected 40 stationary Beta specs, found {len(result)}")
    return result


def stationary_beta_manifest() -> list[dict[str, object]]:
    return [asdict(spec) for spec in stationary_beta_specs()]


def generate_stationary_beta_paths(
    spec: StationaryBetaSpec,
    seed: int,
    paths: int = 1,
) -> dict[str, np.ndarray]:
    """Generate stationary gains and noisy observable routing scores."""
    if paths <= 0:
        raise ValueError("paths must be positive")
    rng = np.random.default_rng(seed)
    shape = (paths, HORIZON, BATCH_SIZE)
    gain = rng.beta(spec.alpha, spec.beta, shape)
    uncertainty = 0.08 + 0.10 * (1.0 - np.abs(2.0 * gain - 1.0))
    score = np.clip(gain + rng.normal(0.0, uncertainty), 0.0, 1.0)
    zeros = np.zeros(shape, dtype=np.float64)
    return {
        "gain": gain,
        "score": score,
        "uncertainty": uncertainty,
        "phase": zeros.copy(),
        "latent": zeros.copy(),
    }
