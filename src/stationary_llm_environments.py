"""Stationary single-subject LLM-derived replay environments.

Every environment draws every query in all 24 batches independently from one
fixed fine-grained MMLU-Pro source pool.  There is no source/category switch,
change point, or batch-bursty sampling.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON


REQUIRED_SPLITS = ("train", "calibration", "test")


@dataclass(frozen=True)
class StationaryLlmSpec:
    environment_id: str
    category: str
    source: str
    subject_label: str
    regime: str
    train_pool_size: int
    calibration_pool_size: int
    test_pool_size: int
    selection_min_pool_size: int
    selection_total_pool_size: int


def subject_label(source: str) -> str:
    """Convert an MMLU-Pro source identifier to a readable plot label."""
    provider, _, raw = source.partition("-")
    provider_labels = {
        "ori_mmlu": "MMLU",
        "stemez": "STEM-EZ",
        "theoremQA": "TheoremQA",
        "scibench": "SciBench",
    }
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw).replace("_", " ")
    words = " ".join(words.split()).title()
    provider_label = provider_labels.get(provider, provider)
    return f"{words} ({provider_label})"


def _slug(source: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    return value[:70]


def stationary_llm_specs(frame: pd.DataFrame, limit: int = 40) -> list[StationaryLlmSpec]:
    """Choose 40 sources without using any reward/outcome information.

    Pools are ranked only by question availability: the smallest of the three
    split-specific pool sizes, followed by total pool size.  This avoids using
    test gains to choose the environments.  Selected sources are then ordered
    by broad category and readable subject name for a stable atlas order.
    """
    counts = pd.crosstab(frame["src"], frame["split"]).reindex(
        columns=list(REQUIRED_SPLITS), fill_value=0
    )
    counts["min_pool_size"] = counts[list(REQUIRED_SPLITS)].min(axis=1)
    counts["total_pool_size"] = counts[list(REQUIRED_SPLITS)].sum(axis=1)
    eligible = counts.loc[counts["min_pool_size"].gt(0)].reset_index()
    eligible = eligible.sort_values(
        ["min_pool_size", "total_pool_size", "src"],
        ascending=[False, False, True],
        kind="stable",
    ).head(limit)
    if len(eligible) != limit:
        raise ValueError(f"expected {limit} eligible sources, found {len(eligible)}")

    source_to_category = (
        frame[["src", "category"]]
        .drop_duplicates()
        .groupby("src", sort=False)["category"]
        .agg(lambda values: str(values.iloc[0]))
    )
    records = []
    for selection_rank, row in enumerate(eligible.itertuples(index=False), 1):
        source = str(row.src)
        records.append({
            "selection_rank": selection_rank,
            "source": source,
            "category": str(source_to_category.loc[source]),
            "subject_label": subject_label(source),
            "train": int(row.train),
            "calibration": int(row.calibration),
            "test": int(row.test),
            "min_pool_size": int(row.min_pool_size),
            "total_pool_size": int(row.total_pool_size),
        })
    ordered = sorted(records, key=lambda x: (x["category"], x["subject_label"], x["source"]))

    result = []
    for position, record in enumerate(ordered, 1):
        result.append(StationaryLlmSpec(
            environment_id=f"llm_stationary_{position:02d}_{_slug(record['source'])}_iid",
            category=record["category"],
            source=record["source"],
            subject_label=record["subject_label"],
            regime="iid_stationary",
            train_pool_size=record["train"],
            calibration_pool_size=record["calibration"],
            test_pool_size=record["test"],
            selection_min_pool_size=record["min_pool_size"],
            selection_total_pool_size=record["total_pool_size"],
        ))
    return result


def sample_stationary_llm_paths(
    frame: pd.DataFrame,
    spec: StationaryLlmSpec,
    split: str,
    seed: int,
    paths: int,
) -> np.ndarray:
    """Bootstrap IID paths from one unchanging subject pool."""
    if split not in REQUIRED_SPLITS:
        raise ValueError(f"unknown split: {split}")
    mask = frame["split"].eq(split) & frame["src"].eq(spec.source)
    candidates = np.flatnonzero(mask.to_numpy())
    if not len(candidates):
        raise ValueError(f"no {split} examples for source={spec.source}")
    rng = np.random.default_rng(seed)
    return rng.choice(
        candidates,
        size=(paths, HORIZON, BATCH_SIZE),
        replace=True,
    )


def stationary_llm_manifest(specs: list[StationaryLlmSpec]) -> list[dict[str, object]]:
    return [asdict(spec) for spec in specs]

