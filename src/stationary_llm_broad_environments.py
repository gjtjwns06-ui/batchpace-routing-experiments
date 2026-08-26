"""Stationary LLM-derived environments based on broad MMLU-Pro categories."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON


REQUIRED_SPLITS = ("train", "calibration", "test")


@dataclass(frozen=True)
class BroadStationaryLlmSpec:
    environment_id: str
    category: str
    category_label: str
    regime: str
    train_pool_size: int
    calibration_pool_size: int
    test_pool_size: int


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def broad_stationary_llm_specs(frame: pd.DataFrame) -> list[BroadStationaryLlmSpec]:
    counts = pd.crosstab(frame["category"], frame["split"]).reindex(
        columns=list(REQUIRED_SPLITS), fill_value=0
    )
    if (counts[list(REQUIRED_SPLITS)] == 0).any(axis=None):
        missing = counts.loc[(counts[list(REQUIRED_SPLITS)] == 0).any(axis=1)]
        raise ValueError(f"broad categories missing required splits:\n{missing}")

    result = []
    for position, (category, row) in enumerate(counts.sort_index().iterrows(), 1):
        category = str(category)
        result.append(BroadStationaryLlmSpec(
            environment_id=f"llm_broad_stationary_{position:02d}_{_slug(category)}_iid",
            category=category,
            category_label=category.title(),
            regime="iid_stationary",
            train_pool_size=int(row["train"]),
            calibration_pool_size=int(row["calibration"]),
            test_pool_size=int(row["test"]),
        ))
    return result


def sample_broad_stationary_paths(
    frame: pd.DataFrame,
    spec: BroadStationaryLlmSpec,
    split: str,
    seed: int,
    paths: int,
) -> np.ndarray:
    if split not in REQUIRED_SPLITS:
        raise ValueError(f"unknown split: {split}")
    mask = frame["split"].eq(split) & frame["category"].eq(spec.category)
    candidates = np.flatnonzero(mask.to_numpy())
    if not len(candidates):
        raise ValueError(f"no {split} examples for category={spec.category}")
    rng = np.random.default_rng(seed)
    return rng.choice(
        candidates,
        size=(paths, HORIZON, BATCH_SIZE),
        replace=True,
    )


def broad_stationary_manifest(specs: list[BroadStationaryLlmSpec]) -> list[dict[str, object]]:
    return [asdict(spec) for spec in specs]

