"""Train routing estimators and replay Main Figure 2 policies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.real_llm_policies import (  # noqa: E402
    PolicyOutcome,
    offline_oracle,
    robust_batch_level,
    wiserouter_offline,
)
from src.synthetic_policies import (  # noqa: E402
    build_batch_value_table,
    build_sequential_value_table,
    run_batchpace,
    run_equal_split,
    run_sequential,
)


CACHE = ROOT / "data" / "real_llm" / "cache" / "mmlu_pro_qwen_pair_1500.parquet"
RESULT_DIR = ROOT / "results" / "real_llm"
BATCH_SIZE = 20
N_TEST = 500
HORIZON = N_TEST // BATCH_SIZE
BUDGET_FRACTIONS = (0.05, 0.10, 0.20)
ORDER_SEEDS = tuple(range(20))
KMEANS_SEEDS = tuple(range(5))
SPLIT_SEEDS = tuple(20260825 + offset for offset in range(5))
N_CLUSTERS = 16
BOOTSTRAPS = 100


def fit_batchpace_predictor(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    probability_columns = [f"cheap_p_{label}" for label in "ABCDEFGHIJ"]
    numeric = probability_columns + [
        "cheap_entropy",
        "cheap_margin",
        "question_length",
        "num_options",
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
        ],
        remainder="drop",
    )
    train_mask = frame["split"].eq("train").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    x_train = preprocessor.fit_transform(frame.loc[train_mask])
    x_all = preprocessor.transform(frame)
    target = frame["gain"].to_numpy(dtype=np.float64)
    model = Ridge(alpha=10.0)
    model.fit(x_train, target[train_mask])
    raw = model.predict(x_all)
    score = np.clip(raw, 0.0, 1.0)
    correlation = spearmanr(raw[test_mask], target[test_mask]).statistic
    diagnostics = {
        "alpha": 10.0,
        "train_mae": float(mean_absolute_error(target[train_mask], raw[train_mask])),
        "test_mae": float(mean_absolute_error(target[test_mask], raw[test_mask])),
        "test_spearman": float(correlation),
        "test_score_mean": float(score[test_mask].mean()),
        "test_score_positive_fraction": float((score[test_mask] > 0).mean()),
    }
    return score, diagnostics


def fit_embedding_space(frame: pd.DataFrame) -> np.ndarray:
    embeddings = np.vstack(frame["embedding"].map(np.asarray))
    train_mask = frame["split"].eq("train").to_numpy()
    pca = PCA(n_components=64, random_state=0)
    reduced = pca.fit_transform(embeddings[train_mask])
    all_reduced = pca.transform(embeddings)
    print(f"Embedding PCA explained variance={pca.explained_variance_ratio_.sum():.4f}", flush=True)
    return normalize(all_reduced)


def robust_lower_bounds(
    frame: pd.DataFrame, embedding_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_mask = frame["split"].eq("train").to_numpy()
    category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    category_train = category_encoder.fit_transform(frame.loc[train_mask, ["category"]])
    category_all = category_encoder.transform(frame[["category"]])
    basic = np.column_stack(
        [
            np.log1p(frame["question_length"].to_numpy(dtype=np.float64)),
            frame["num_options"].to_numpy(dtype=np.float64),
        ]
    )
    x_all = np.column_stack([embedding_features, category_all, basic])
    x_train = x_all[train_mask]
    rng = np.random.default_rng(20260825)
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, float] = {}
    for target_name in ("cheap_correct", "strong_correct"):
        target = frame[target_name].to_numpy(dtype=np.float64)
        bootstrap_predictions = np.empty((BOOTSTRAPS, len(frame)), dtype=np.float32)
        for bootstrap in range(BOOTSTRAPS):
            sampled = rng.integers(0, len(x_train), len(x_train))
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=2.0,
                random_state=bootstrap,
            )
            model.fit(x_train[sampled], target[train_mask][sampled])
            bootstrap_predictions[bootstrap] = np.clip(model.predict(x_all), 0.0, 1.0)
        predictions[target_name] = np.quantile(bootstrap_predictions, 0.10, axis=0)
        mean_prediction = bootstrap_predictions.mean(axis=0)
        test_mask = frame["split"].eq("test").to_numpy()
        diagnostics[f"{target_name}_test_mae"] = float(
            mean_absolute_error(target[test_mask], mean_prediction[test_mask])
        )
        diagnostics[f"{target_name}_mean_interval_width"] = float(
            (
                np.quantile(bootstrap_predictions, 0.95, axis=0)
                - np.quantile(bootstrap_predictions, 0.05, axis=0)
            ).mean()
        )
        print(f"Robust bootstrap complete: {target_name}", flush=True)
    diagnostics["bootstrap_resamples"] = BOOTSTRAPS
    diagnostics["lower_quantile"] = 0.10
    return predictions["cheap_correct"], predictions["strong_correct"], diagnostics


def fit_wiserouter_contexts(
    frame: pd.DataFrame, embedding_features: np.ndarray
) -> list[dict[str, np.ndarray]]:
    train_mask = frame["split"].eq("train").to_numpy()
    results = []
    for seed in KMEANS_SEEDS:
        model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
        model.fit(embedding_features[train_mask])
        contexts = model.predict(embedding_features)
        train_contexts = contexts[train_mask]
        counts = np.bincount(train_contexts, minlength=N_CLUSTERS)
        pi = counts / counts.sum()
        cheap = np.zeros(N_CLUSTERS)
        strong = np.zeros(N_CLUSTERS)
        for context in range(N_CLUSTERS):
            members = train_mask & (contexts == context)
            if not members.any():
                raise AssertionError("K-means produced an empty train context")
            cheap[context] = frame.loc[members, "cheap_correct"].mean()
            strong[context] = frame.loc[members, "strong_correct"].mean()
        results.append({"contexts": contexts, "pi": pi, "cheap": cheap, "strong": strong})
    return results


def calibration_batches(
    frame: pd.DataFrame, scores: np.ndarray, environment: str, seed: int
) -> np.ndarray:
    calibration_indices = np.flatnonzero(frame["split"].eq("calibration").to_numpy())
    rng = np.random.default_rng(seed)
    if environment == "random":
        sampled = rng.choice(calibration_indices, size=(1000, BATCH_SIZE), replace=True)
        return scores[sampled]

    categories = frame.loc[calibration_indices, "category"].to_numpy()
    names, counts = np.unique(categories, return_counts=True)
    probabilities = counts / counts.sum()
    batches = np.empty((1000, BATCH_SIZE), dtype=np.float64)
    for batch in range(1000):
        category = rng.choice(names, p=probabilities)
        candidates = calibration_indices[categories == category]
        batches[batch] = scores[rng.choice(candidates, BATCH_SIZE, replace=True)]
    return batches


def test_order(frame: pd.DataFrame, environment: str, seed: int) -> np.ndarray:
    test_indices = np.flatnonzero(frame["split"].eq("test").to_numpy())
    test_indices = test_indices[np.argsort(frame.loc[test_indices, "question_id"].to_numpy())]
    rng = np.random.default_rng(seed)
    if environment == "random":
        return test_indices[rng.permutation(len(test_indices))]

    categories = frame.loc[test_indices, "category"].to_numpy()
    category_order = np.unique(categories)
    rng.shuffle(category_order)
    ordered: list[int] = []
    for category in category_order:
        members = test_indices[categories == category].copy()
        rng.shuffle(members)
        ordered.extend(int(value) for value in members)
    return np.asarray(ordered, dtype=np.int64)


def add_record(
    records: list[dict[str, object]],
    *,
    method: str,
    split_seed: int,
    environment: str,
    order_seed: int,
    budget_fraction: float,
    outcome: PolicyOutcome,
    oracle: PolicyOutcome,
    cheap_correct: np.ndarray,
) -> None:
    if outcome.budget_used > round(budget_fraction * N_TEST):
        raise AssertionError(f"Budget violation for {method}")
    if outcome.true_gain > oracle.true_gain + 1e-9:
        raise AssertionError(f"{method} exceeds offline oracle")
    accuracy = (float(cheap_correct.sum()) + outcome.true_gain) / N_TEST
    recovery = outcome.true_gain / oracle.true_gain if oracle.true_gain > 0 else np.nan
    records.append(
        {
            "method": method,
            "split_seed": split_seed,
            "environment": environment,
            "order_seed": order_seed,
            "budget_fraction": budget_fraction,
            "budget": round(budget_fraction * N_TEST),
            "accuracy": accuracy,
            "delta_accuracy_pp": 100.0 * outcome.true_gain / N_TEST,
            "true_gain": outcome.true_gain,
            "oracle_gain": oracle.true_gain,
            "oracle_recovery": recovery,
            "budget_used": outcome.budget_used,
        }
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    within_split = (
        metrics.groupby(
            ["split_seed", "environment", "budget_fraction", "method"], sort=True
        )[
            [
                "accuracy",
                "delta_accuracy_pp",
                "oracle_recovery",
                "budget_used",
            ]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for keys, group in within_split.groupby(["environment", "budget_fraction", "method"], sort=True):
        environment, budget_fraction, method = keys
        n = group["split_seed"].nunique()
        rows.append(
            {
                "environment": environment,
                "budget_fraction": budget_fraction,
                "budget_percent": 100 * budget_fraction,
                "method": method,
                "split_runs": n,
                "order_seeds_per_split": len(ORDER_SEEDS),
                "mean_accuracy": group["accuracy"].mean(),
                "ci95_accuracy": 1.96 * group["accuracy"].std(ddof=1) / np.sqrt(n),
                "mean_delta_accuracy_pp": group["delta_accuracy_pp"].mean(),
                "ci95_delta_accuracy_pp": 1.96 * group["delta_accuracy_pp"].std(ddof=1) / np.sqrt(n),
                "mean_oracle_recovery": group["oracle_recovery"].mean(),
                "mean_budget_used": group["budget_used"].mean(),
            }
        )
    return pd.DataFrame(rows)


def assign_repeated_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = np.arange(len(frame))
    train_indices, remainder = train_test_split(
        indices,
        train_size=500,
        random_state=seed,
        stratify=frame["category"],
    )
    calibration_indices, test_indices = train_test_split(
        remainder,
        train_size=500,
        random_state=seed + 10_000,
        stratify=frame.loc[remainder, "category"],
    )
    result = frame.copy()
    result["split"] = ""
    result.loc[train_indices, "split"] = "train"
    result.loc[calibration_indices, "split"] = "calibration"
    result.loc[test_indices, "split"] = "test"
    if result["split"].value_counts().to_dict() != {"train": 500, "calibration": 500, "test": 500}:
        raise AssertionError("Repeated split must contain 500 rows per partition")
    return result


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base_frame = pd.read_parquet(CACHE)
    records: list[dict[str, object]] = []
    diagnostics_by_split = []
    prediction_arrays = []
    for split_position, split_seed in enumerate(SPLIT_SEEDS):
        print(f"\n=== Split seed {split_seed} ({split_position + 1}/{len(SPLIT_SEEDS)}) ===", flush=True)
        frame = assign_repeated_split(base_frame, split_seed)
        split_ids = {
            split: set(frame.loc[frame["split"].eq(split), "question_id"])
            for split in ("train", "calibration", "test")
        }
        if any(split_ids[a] & split_ids[b] for a in split_ids for b in split_ids if a < b):
            raise AssertionError("Split leakage detected")

        batchpace_score, ridge_diagnostics = fit_batchpace_predictor(frame)
        print(json.dumps(ridge_diagnostics, indent=2), flush=True)
        embedding_features = fit_embedding_space(frame)
        robust_cheap, robust_strong, robust_diagnostics = robust_lower_bounds(
            frame, embedding_features
        )
        robust_increment = robust_strong - robust_cheap
        wiserouter_models = fit_wiserouter_contexts(frame, embedding_features)
        prediction_arrays.append((batchpace_score, robust_cheap, robust_strong, robust_increment))
        diagnostics_by_split.append(
            {
                "split_seed": split_seed,
                "ridge": ridge_diagnostics,
                "robust_estimator": robust_diagnostics,
            }
        )

        calibration_tables: dict[tuple[str, int], np.ndarray] = {}
        sequential_tables: dict[int, np.ndarray] = {}
        for environment, seed in (("random", 3101), ("category-bursty", 3102)):
            batches = calibration_batches(frame, batchpace_score, environment, seed + split_position)
            for fraction in BUDGET_FRACTIONS:
                budget = round(fraction * N_TEST)
                calibration_tables[(environment, budget)] = build_batch_value_table(
                    batches, horizon=HORIZON, budget=budget
                )
        calibration_scores = batchpace_score[frame["split"].eq("calibration").to_numpy()]
        for fraction in BUDGET_FRACTIONS:
            budget = round(fraction * N_TEST)
            sequential_tables[budget] = build_sequential_value_table(
                calibration_scores.reshape(-1, 1), horizon=N_TEST, budget=budget
            )

        for environment in ("random", "category-bursty"):
            for order_seed in ORDER_SEEDS:
                order = test_order(frame, environment, order_seed)
                true_gain = frame.loc[order, "gain"].to_numpy(dtype=np.float64)
                cheap_correct = frame.loc[order, "cheap_correct"].to_numpy(dtype=np.float64)
                scores = batchpace_score[order]
                robust_scores = robust_increment[order]
                wr = wiserouter_models[order_seed % len(wiserouter_models)]
                contexts = wr["contexts"][order]
                for fraction in BUDGET_FRACTIONS:
                    budget = round(fraction * N_TEST)
                    oracle = offline_oracle(true_gain, budget, BATCH_SIZE)
                    outcomes: dict[str, PolicyOutcome] = {
                        "Offline Oracle": oracle,
                        "BatchPace": run_batchpace(
                            scores, true_gain, BATCH_SIZE, budget,
                            calibration_tables[(environment, budget)],
                        ),
                        "Sequential": run_sequential(
                            scores, true_gain, budget, sequential_tables[budget], BATCH_SIZE,
                        ),
                        "Equal Split": run_equal_split(scores, true_gain, BATCH_SIZE, budget),
                        "Robust Batch-Level": robust_batch_level(
                            robust_scores, true_gain, BATCH_SIZE, budget
                        ),
                        "WISERouter-Offline": wiserouter_offline(
                            contexts, wr["pi"], wr["cheap"], wr["strong"], true_gain,
                            budget, BATCH_SIZE,
                            seed=(
                                900_000 + 100_000 * split_position + 1000 * order_seed
                                + budget + (0 if environment == "random" else 50_000)
                            ),
                        ),
                    }
                    for method, outcome in outcomes.items():
                        add_record(
                            records,
                            method=method,
                            split_seed=split_seed,
                            environment=environment,
                            order_seed=order_seed,
                            budget_fraction=fraction,
                            outcome=outcome,
                            oracle=oracle,
                            cheap_correct=cheap_correct,
                        )

    metrics = pd.DataFrame(records)
    summary = summarize(metrics)
    metrics.to_csv(RESULT_DIR / "policy_metrics.csv", index=False)
    summary.to_csv(RESULT_DIR / "policy_summary.csv", index=False)
    summary.loc[summary["budget_fraction"].eq(0.10)].to_csv(
        RESULT_DIR / "main_table_10pct.csv", index=False
    )
    np.savez_compressed(
        RESULT_DIR / "policy_prediction_arrays.npz",
        batchpace_score=np.stack([value[0] for value in prediction_arrays]),
        robust_cheap_lower=np.stack([value[1] for value in prediction_arrays]),
        robust_strong_lower=np.stack([value[2] for value in prediction_arrays]),
        robust_increment=np.stack([value[3] for value in prediction_arrays]),
    )
    diagnostics = {
        "leakage_check": "passed",
        "split_diagnostics": diagnostics_by_split,
        "protocol": {
            "batch_size": BATCH_SIZE,
            "test_queries": N_TEST,
            "order_seeds": len(ORDER_SEEDS),
            "split_seeds": list(SPLIT_SEEDS),
            "wiserouter_clusters": N_CLUSTERS,
            "wiserouter_kmeans_seeds": len(KMEANS_SEEDS),
            "budgets": list(BUDGET_FRACTIONS),
            "robust_per_batch_caps": [round(value * BATCH_SIZE) for value in BUDGET_FRACTIONS],
        },
    }
    (RESULT_DIR / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
