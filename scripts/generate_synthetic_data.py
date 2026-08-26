"""Generate reproducible query-level data for Main Figure 1.

The synthetic queries are numerical records, not natural-language prompts.  Each
record contains a latent true gain and a unit noise draw.  A predicted score for
any error level epsilon is recovered as

    clip(true_gain + epsilon * noise_unit, 0, 1).

Keeping the unit noise draw rather than materializing every epsilon makes the
dataset smaller and makes paired error sweeps exact and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT_SEED = 20260825
N_QUERIES = 1_000
N_EPISODES = 250
BASE_BATCH_SIZE = 20
CALIBRATION_BATCHES = 1_000
BUDGET = 50
HIGH_PROBABILITY = 0.20
LOW_BETA = (1.0, 12.0)
HIGH_BETA = (5.0, 2.0)
EPSILONS = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
BATCH_SIZES = (1, 5, 10, 20, 50, 100)
BATCH_SWEEP_EPSILON = 0.10
ENVIRONMENTS = ("iid", "bursty")


def deterministic_seed(tag: str) -> int:
    payload = f"{ROOT_SEED}:{tag}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng(deterministic_seed(tag))


def generate_queries(
    *,
    environment: str,
    records: int,
    queries_per_record: int,
    batch_size: int,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true_gain, high_state, and unit_noise arrays.

    ``records`` means test episodes for test data and independent batches for
    calibration data.  For calibration data, ``queries_per_record`` equals the
    batch size.
    """
    if queries_per_record % batch_size != 0:
        raise ValueError("queries_per_record must be divisible by batch_size")

    rng = rng_for(tag)
    shape = (records, queries_per_record)

    if environment == "iid":
        high_state = rng.random(shape) < HIGH_PROBABILITY
    elif environment == "bursty":
        batches_per_record = queries_per_record // batch_size
        batch_state = rng.random((records, batches_per_record)) < HIGH_PROBABILITY
        high_state = np.repeat(batch_state, batch_size, axis=1)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    low_gain = rng.beta(*LOW_BETA, size=shape)
    high_gain = rng.beta(*HIGH_BETA, size=shape)
    true_gain = np.where(high_state, high_gain, low_gain).astype(np.float32)
    unit_noise = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    return true_gain, high_state.astype(np.uint8), unit_noise


def build_npz_payload(
    *,
    split: str,
    batch_sizes: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    arrays: dict[str, np.ndarray] = {}
    seeds: dict[str, int] = {}

    for batch_size in batch_sizes:
        records = N_EPISODES if split == "test" else CALIBRATION_BATCHES
        queries_per_record = N_QUERIES if split == "test" else batch_size

        for environment in ENVIRONMENTS:
            tag = f"{split}:m={batch_size}:{environment}"
            true_gain, high_state, unit_noise = generate_queries(
                environment=environment,
                records=records,
                queries_per_record=queries_per_record,
                batch_size=batch_size,
                tag=tag,
            )
            prefix = f"m{batch_size:03d}_{environment}"
            arrays[f"{prefix}_true_gain"] = true_gain
            arrays[f"{prefix}_high_state"] = high_state
            arrays[f"{prefix}_unit_noise"] = unit_noise
            seeds[tag] = deterministic_seed(tag)

    return arrays, seeds


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_from_base(true_gain: np.ndarray, unit_noise: np.ndarray, epsilon: float) -> np.ndarray:
    return np.clip(true_gain + epsilon * unit_noise, 0.0, 1.0)


def write_previews(output_dir: Path, fig1a: dict[str, np.ndarray], fig1b: dict[str, np.ndarray]) -> None:
    preview_a = output_dir / "fig1a_preview.jsonl"
    with preview_a.open("w", encoding="utf-8", newline="\n") as stream:
        for environment in ENVIRONMENTS:
            prefix = f"m{BASE_BATCH_SIZE:03d}_{environment}"
            gain = fig1a[f"{prefix}_true_gain"]
            state = fig1a[f"{prefix}_high_state"]
            noise = fig1a[f"{prefix}_unit_noise"]
            for query_id in range(12):
                record = {
                    "panel": "fig1a_prediction_error",
                    "environment": environment,
                    "episode": 0,
                    "batch_size": BASE_BATCH_SIZE,
                    "batch_id": query_id // BASE_BATCH_SIZE,
                    "query_in_batch": query_id % BASE_BATCH_SIZE,
                    "high_state": int(state[0, query_id]),
                    "true_gain": round(float(gain[0, query_id]), 8),
                    "unit_noise": round(float(noise[0, query_id]), 8),
                    "predicted_scores": {
                        f"eps_{epsilon:.2f}": round(
                            float(np.clip(gain[0, query_id] + epsilon * noise[0, query_id], 0.0, 1.0)),
                            8,
                        )
                        for epsilon in EPSILONS
                    },
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    preview_b = output_dir / "fig1b_preview.jsonl"
    with preview_b.open("w", encoding="utf-8", newline="\n") as stream:
        for batch_size in BATCH_SIZES:
            for environment in ENVIRONMENTS:
                prefix = f"m{batch_size:03d}_{environment}"
                gain = fig1b[f"{prefix}_true_gain"]
                state = fig1b[f"{prefix}_high_state"]
                noise = fig1b[f"{prefix}_unit_noise"]
                score = score_from_base(gain, noise, BATCH_SWEEP_EPSILON)
                for query_id in range(min(3, N_QUERIES)):
                    record = {
                        "panel": "fig1b_batch_size",
                        "environment": environment,
                        "episode": 0,
                        "batch_size": batch_size,
                        "batch_id": query_id // batch_size,
                        "query_in_batch": query_id % batch_size,
                        "high_state": int(state[0, query_id]),
                        "true_gain": round(float(gain[0, query_id]), 8),
                        "unit_noise": round(float(noise[0, query_id]), 8),
                        "epsilon": BATCH_SWEEP_EPSILON,
                        "predicted_score": round(float(score[0, query_id]), 8),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def environment_summary(arrays: dict[str, np.ndarray], batch_size: int, environment: str) -> dict[str, Any]:
    prefix = f"m{batch_size:03d}_{environment}"
    gain = arrays[f"{prefix}_true_gain"]
    state = arrays[f"{prefix}_high_state"]
    noise = arrays[f"{prefix}_unit_noise"]
    reshaped_state = state.reshape(state.shape[0], -1, batch_size)
    constant_batches = np.all(reshaped_state == reshaped_state[:, :, :1], axis=2)
    return {
        "shape": list(gain.shape),
        "high_rate": float(state.mean()),
        "mean_true_gain": float(gain.mean()),
        "min_true_gain": float(gain.min()),
        "max_true_gain": float(gain.max()),
        "min_unit_noise": float(noise.min()),
        "max_unit_noise": float(noise.max()),
        "constant_state_batch_rate": float(constant_batches.mean()),
    }


def validate_dataset(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    expected_rows = N_EPISODES if split == "test" else CALIBRATION_BATCHES

    for batch_size in batch_sizes:
        expected_cols = N_QUERIES if split == "test" else batch_size
        for environment in ENVIRONMENTS:
            prefix = f"m{batch_size:03d}_{environment}"
            gain = arrays[f"{prefix}_true_gain"]
            state = arrays[f"{prefix}_high_state"]
            noise = arrays[f"{prefix}_unit_noise"]

            expected_shape = (expected_rows, expected_cols)
            if gain.shape != expected_shape or state.shape != expected_shape or noise.shape != expected_shape:
                raise AssertionError(f"Unexpected shape for {split}/{prefix}")
            if not np.all((gain >= 0.0) & (gain <= 1.0)):
                raise AssertionError(f"Gain outside [0, 1] for {split}/{prefix}")
            if not np.all((noise >= -1.0) & (noise <= 1.0)):
                raise AssertionError(f"Noise outside [-1, 1] for {split}/{prefix}")

            batch_view = state.reshape(expected_rows, -1, batch_size)
            constant = np.all(batch_view == batch_view[:, :, :1], axis=2)
            if environment == "bursty" and not bool(np.all(constant)):
                raise AssertionError(f"Bursty batch state is not constant for {split}/{prefix}")

            report[f"m={batch_size}/{environment}"] = environment_summary(
                arrays, batch_size, environment
            )

    return report


def write_manifest(
    output_dir: Path,
    files: list[Path],
    seeds: dict[str, int],
    validation: dict[str, Any],
) -> None:
    manifest = {
        "dataset": "BatchPace Main Figure 1 synthetic controlled study",
        "schema_version": 1,
        "root_seed": ROOT_SEED,
        "parameters": {
            "n_queries_per_test_episode": N_QUERIES,
            "test_episodes": N_EPISODES,
            "calibration_batches": CALIBRATION_BATCHES,
            "budget": BUDGET,
            "base_batch_size": BASE_BATCH_SIZE,
            "base_n_batches": N_QUERIES // BASE_BATCH_SIZE,
            "high_probability": HIGH_PROBABILITY,
            "low_gain_beta": list(LOW_BETA),
            "high_gain_beta": list(HIGH_BETA),
            "prediction_error_epsilons": list(EPSILONS),
            "batch_sizes": list(BATCH_SIZES),
            "batch_sweep_epsilon": BATCH_SWEEP_EPSILON,
        },
        "score_definition": "predicted_score = clip(true_gain + epsilon * unit_noise, 0, 1)",
        "array_key_pattern": "m{batch_size:03d}_{environment}_{true_gain|high_state|unit_noise}",
        "index_semantics": {
            "test": "array[episode, query_id]; batch_id = query_id // batch_size",
            "calibration": "array[calibration_batch_id, query_in_batch]",
        },
        "random_streams": seeds,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "validation": validation,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_main_figure_1"),
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1a_test, seeds_a_test = build_npz_payload(
        split="test", batch_sizes=(BASE_BATCH_SIZE,)
    )
    fig1a_calibration, seeds_a_cal = build_npz_payload(
        split="calibration", batch_sizes=(BASE_BATCH_SIZE,)
    )
    fig1b_test, seeds_b_test = build_npz_payload(
        split="test", batch_sizes=BATCH_SIZES
    )
    fig1b_calibration, seeds_b_cal = build_npz_payload(
        split="calibration", batch_sizes=BATCH_SIZES
    )

    paths = [
        output_dir / "fig1a_test.npz",
        output_dir / "fig1a_calibration.npz",
        output_dir / "fig1b_test.npz",
        output_dir / "fig1b_calibration.npz",
    ]
    save_npz(paths[0], fig1a_test)
    save_npz(paths[1], fig1a_calibration)
    save_npz(paths[2], fig1b_test)
    save_npz(paths[3], fig1b_calibration)
    write_previews(output_dir, fig1a_test, fig1b_test)

    validation = {
        "fig1a_test": validate_dataset(
            fig1a_test, split="test", batch_sizes=(BASE_BATCH_SIZE,)
        ),
        "fig1a_calibration": validate_dataset(
            fig1a_calibration,
            split="calibration",
            batch_sizes=(BASE_BATCH_SIZE,),
        ),
        "fig1b_test": validate_dataset(
            fig1b_test, split="test", batch_sizes=BATCH_SIZES
        ),
        "fig1b_calibration": validate_dataset(
            fig1b_calibration,
            split="calibration",
            batch_sizes=BATCH_SIZES,
        ),
    }

    preview_paths = [output_dir / "fig1a_preview.jsonl", output_dir / "fig1b_preview.jsonl"]
    all_seeds = {**seeds_a_test, **seeds_a_cal, **seeds_b_test, **seeds_b_cal}
    write_manifest(output_dir, paths + preview_paths, all_seeds, validation)

    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "files": [path.name for path in paths + preview_paths] + ["manifest.json"],
        "fig1a_bursty_high_rate": validation["fig1a_test"]["m=20/bursty"]["high_rate"],
        "fig1a_iid_high_rate": validation["fig1a_test"]["m=20/iid"]["high_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
