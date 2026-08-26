"""Run the 40 synthetic and 40 LLM-derived BatchPace environments."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark_suite_environments import (  # noqa: E402
    BATCH_SIZE,
    HORIZON,
    SHIFT_BATCH,
    SyntheticSpec,
    generate_synthetic_paths,
    synthetic_manifest,
    synthetic_specs,
)
from src.benchmark_suite_policies import (  # noqa: E402
    Outcome,
    build_batch_dp,
    build_sequential_dp,
    offline_oracle,
    run_batchpace,
    run_equal_split,
    run_fixed_threshold,
    run_robust_batch,
    run_sequential,
    run_wiserouter,
    threshold_from_calibration,
)


DEFAULT_BUDGETS = (0, 24, 48, 72, 96, 120, 144, 192)
METHODS = (
    "BatchPace",
    "Oracle BatchPace",
    "Sequential Pacing",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)


@dataclass(frozen=True)
class LlmSpec:
    environment_id: str
    pre_category: str
    post_category: str
    regime: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("synthetic", "llm", "all"), default="all")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--calibration-paths", type=int, default=200)
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--cheap-scores", type=Path,
                   default=ROOT / "results/benchmark_suite/option_scores_qwen3_0.6b.jsonl")
    p.add_argument("--strong-scores", type=Path,
                   default=ROOT / "results/benchmark_suite/option_scores_qwen3_4b.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "results/benchmark_suite")
    return p.parse_args()


def check_outcome(method: str, outcome: Outcome, oracle: Outcome, budget: int) -> None:
    if outcome.budget_used > budget:
        raise AssertionError(f"{method} used {outcome.budget_used}>{budget}")
    if outcome.gain > oracle.gain + 1e-9:
        raise AssertionError(f"{method} exceeds hindsight oracle")


def record(rows: list[dict[str, object]], environment_type: str, environment_id: str,
           seed: int, budget: int, method: str, outcome: Outcome) -> None:
    rows.append({
        "environment_type": environment_type,
        "environment_id": environment_id,
        "seed": seed,
        "budget": budget,
        "budget_fraction": budget / (HORIZON * BATCH_SIZE),
        "method": method,
        "total_realized_gain": outcome.gain,
        "budget_used": outcome.budget_used,
        "batch_allocations": json.dumps(outcome.allocations),
    })


def evaluate_methods(*, gain: np.ndarray, score: np.ndarray, robust_score: np.ndarray,
                     contexts: np.ndarray, pi: np.ndarray, cheap_reward: np.ndarray,
                     strong_reward: np.ndarray, calibration_score: np.ndarray,
                     calibration_gain: np.ndarray, budgets: list[int], seed: int,
                     environment_type: str, environment_id: str,
                     rows: list[dict[str, object]],
                     calibration_artifacts: dict[str, object] | None = None) -> None:
    if calibration_artifacts is None:
        max_budget = max(budgets)
        calibration_artifacts = {
            "score_batch_dp": build_batch_dp(calibration_score, max_budget),
            "gain_batch_dp": build_batch_dp(calibration_gain, max_budget),
            "sequential_dp": build_sequential_dp(calibration_score, max_budget),
            "thresholds": {
                budget: threshold_from_calibration(calibration_score, budget)
                for budget in budgets
            },
        }
    for budget in budgets:
        hindsight = offline_oracle(gain, budget, BATCH_SIZE)
        score_batch_dp = calibration_artifacts["score_batch_dp"]
        gain_batch_dp = calibration_artifacts["gain_batch_dp"]
        sequential_dp = calibration_artifacts["sequential_dp"]
        threshold = calibration_artifacts["thresholds"][budget]
        outcomes = {
            "BatchPace": run_batchpace(score, gain, BATCH_SIZE, budget, score_batch_dp),
            "Oracle BatchPace": run_batchpace(gain, gain, BATCH_SIZE, budget, gain_batch_dp),
            "Sequential Pacing": run_sequential(
                score, gain, BATCH_SIZE, budget, sequential_dp
            ),
            "Fixed Threshold": run_fixed_threshold(
                score, gain, BATCH_SIZE, budget, threshold
            ),
            "Equal Split": run_equal_split(score, gain, BATCH_SIZE, budget),
            "WISERouter-Offline": run_wiserouter(
                contexts, pi, cheap_reward, strong_reward, gain, BATCH_SIZE, budget,
                seed=8_000_000 + seed * 1000 + budget,
            ),
            "Robust Batch-Level Query Routing": run_robust_batch(
                robust_score, gain, BATCH_SIZE, budget
            ),
        }
        for method in METHODS:
            check_outcome(method, outcomes[method], hindsight, budget)
            record(rows, environment_type, environment_id, seed, budget, method,
                   outcomes[method])


def synthetic_context_model(spec: SyntheticSpec) -> tuple[KMeans, np.ndarray, np.ndarray, np.ndarray]:
    stable = zlib.crc32(spec.environment_id.encode("utf-8"))
    train = generate_synthetic_paths(spec, 1_000_000 + stable % 100_000, 30)
    features = np.column_stack([
        train["score"].ravel(),
        train["uncertainty"].ravel(),
        train["phase"].ravel(),
        train["latent"].ravel(),
    ])
    model = KMeans(n_clusters=16, n_init=10, random_state=0).fit(features)
    context = model.labels_
    gain = train["gain"].ravel()
    counts = np.bincount(context, minlength=16)
    pi = counts / counts.sum()
    cheap = np.zeros(16, dtype=np.float64)
    strong = np.zeros(16, dtype=np.float64)
    for j in range(16):
        strong[j] = gain[context == j].mean()
    return model, pi, cheap, strong


def run_synthetic(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest = synthetic_manifest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic_environment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for position, spec in enumerate(synthetic_specs(), 1):
        print(f"[synthetic {position:02d}/40] {spec.environment_id}", flush=True)
        calibration = generate_synthetic_paths(
            spec, 2_000_000 + position, args.calibration_paths
        )
        calibration_artifacts = {
            "score_batch_dp": build_batch_dp(calibration["score"], max(args.budgets)),
            "gain_batch_dp": build_batch_dp(calibration["gain"], max(args.budgets)),
            "sequential_dp": build_sequential_dp(calibration["score"], max(args.budgets)),
            "thresholds": {
                budget: threshold_from_calibration(calibration["score"], budget)
                for budget in args.budgets
            },
        }
        context_model, pi, cheap_reward, strong_reward = synthetic_context_model(spec)
        for seed in range(args.seeds):
            data = generate_synthetic_paths(spec, 3_000_000 + position * 1000 + seed, 1)
            gain = data["gain"].reshape(-1)
            score = data["score"].reshape(-1)
            features = np.column_stack([
                score,
                data["uncertainty"].reshape(-1),
                data["phase"].reshape(-1),
                data["latent"].reshape(-1),
            ])
            contexts = context_model.predict(features)
            robust = score - 1.2815515655446004 * data["uncertainty"].reshape(-1)
            evaluate_methods(
                gain=gain,
                score=score,
                robust_score=robust,
                contexts=contexts,
                pi=pi,
                cheap_reward=cheap_reward,
                strong_reward=strong_reward,
                calibration_score=calibration["score"],
                calibration_gain=calibration["gain"],
                budgets=args.budgets,
                seed=seed,
                environment_type="synthetic",
                environment_id=spec.environment_id,
                rows=rows,
                calibration_artifacts=calibration_artifacts,
            )
    result = pd.DataFrame(rows)
    result.to_parquet(args.output / "synthetic_metrics.parquet", index=False)
    result.drop(columns="batch_allocations").to_csv(
        args.output / "synthetic_metrics.csv", index=False
    )
    return result


def read_jsonl(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing option-score cache: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = pd.DataFrame(rows)
    if result["question_id"].duplicated().any():
        raise ValueError(f"duplicate question_id in {path}")
    probability = pd.DataFrame(result["option_probability"].tolist()).fillna(0.0)
    probability = probability.reindex(columns=list("ABCDEFGHIJ"), fill_value=0.0)
    probability.columns = [f"{prefix}_p_{x}" for x in probability.columns]
    keep = result[["question_id", "correct", "entropy", "margin"]].rename(columns={
        "correct": f"{prefix}_correct",
        "entropy": f"{prefix}_entropy",
        "margin": f"{prefix}_margin",
    })
    return pd.concat([keep.reset_index(drop=True), probability.reset_index(drop=True)], axis=1)


def load_llm_frame(cheap_path: Path, strong_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "data/real_llm/splits/mmlu_pro_main_1500.parquet")
    cheap = read_jsonl(cheap_path, "cheap")
    strong = read_jsonl(strong_path, "strong")
    frame = base.merge(cheap, on="question_id", validate="one_to_one")
    frame = frame.merge(strong, on="question_id", validate="one_to_one")
    if len(frame) != len(base):
        raise ValueError(f"option-score caches incomplete: merged {len(frame)}/{len(base)}")
    frame["gain"] = frame["strong_correct"] - frame["cheap_correct"]
    frame["question_length"] = frame["question"].str.len()
    frame["num_options"] = frame["options"].map(len)
    embedding_source = pd.read_parquet(
        ROOT / "data/real_llm/cache/mmlu_pro_qwen_pair_1500.parquet",
        columns=["question_id", "embedding"],
    )
    frame = frame.merge(embedding_source, on="question_id", validate="one_to_one")
    return frame


def fit_llm_predictor(frame: pd.DataFrame) -> np.ndarray:
    probability_columns = [f"cheap_p_{x}" for x in "ABCDEFGHIJ"]
    numeric = probability_columns + [
        "cheap_entropy", "cheap_margin", "question_length", "num_options"
    ]
    transform = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["category"]),
    ])
    train = frame["split"].eq("train").to_numpy()
    x_train = transform.fit_transform(frame.loc[train])
    x_all = transform.transform(frame)
    model = Ridge(alpha=10.0).fit(x_train, frame.loc[train, "gain"])
    return np.clip(model.predict(x_all), -1.0, 1.0)


def embedding_features(frame: pd.DataFrame) -> np.ndarray:
    embeddings = np.vstack(frame["embedding"].map(np.asarray))
    train = frame["split"].eq("train").to_numpy()
    components = min(64, embeddings.shape[1], int(train.sum()) - 1)
    pca = PCA(n_components=components, random_state=0).fit(embeddings[train])
    return normalize(pca.transform(embeddings))


def fit_llm_contexts(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = frame["split"].eq("train").to_numpy()
    model = KMeans(n_clusters=16, n_init=10, random_state=0).fit(embeddings[train])
    contexts = model.predict(embeddings)
    counts = np.bincount(contexts[train], minlength=16)
    pi = counts / counts.sum()
    cheap = np.zeros(16)
    strong = np.zeros(16)
    for j in range(16):
        members = train & (contexts == j)
        cheap[j] = frame.loc[members, "cheap_correct"].mean()
        strong[j] = frame.loc[members, "strong_correct"].mean()
    return contexts, pi, cheap, strong


def fit_robust_scores(frame: pd.DataFrame, embeddings: np.ndarray, bootstraps: int = 100) -> np.ndarray:
    """10th-percentile bootstrap incremental reward used by robust routing."""
    train = frame["split"].eq("train").to_numpy()
    categories = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    category_train = categories.fit_transform(frame.loc[train, ["category"]])
    category_all = categories.transform(frame[["category"]])
    basic = np.column_stack([
        np.log1p(frame["question_length"].to_numpy()),
        frame["num_options"].to_numpy(),
        frame["cheap_entropy"].to_numpy(),
        frame["cheap_margin"].to_numpy(),
    ])
    x_all = np.column_stack([embeddings, category_all, basic])
    x_train = x_all[train]
    rng = np.random.default_rng(20260825)
    lower: dict[str, np.ndarray] = {}
    for target_name in ("cheap_correct", "strong_correct"):
        target = frame[target_name].to_numpy(dtype=np.float64)
        predictions = np.empty((bootstraps, len(frame)), dtype=np.float32)
        for bootstrap in range(bootstraps):
            sampled = rng.integers(0, len(x_train), len(x_train))
            model = HistGradientBoostingRegressor(
                learning_rate=0.06, max_iter=60, max_leaf_nodes=15,
                min_samples_leaf=15, l2_regularization=2.0,
                random_state=bootstrap,
            ).fit(x_train[sampled], target[train][sampled])
            predictions[bootstrap] = np.clip(model.predict(x_all), 0, 1)
        lower[target_name] = np.quantile(predictions, 0.10, axis=0)
        print(f"robust bootstrap complete: {target_name}", flush=True)
    return lower["strong_correct"] - lower["cheap_correct"]


def make_llm_specs(frame: pd.DataFrame) -> list[LlmSpec]:
    calibration = frame.loc[frame["split"].eq("calibration")]
    mean = calibration.groupby("category")["gain"].mean()
    pairs: list[tuple[float, str, str]] = []
    names = sorted(mean.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((abs(float(mean[a] - mean[b])), str(a), str(b)))
    selected = sorted(pairs, reverse=True)[:10]
    directions = [(a, b) for _, a, b in selected] + [(b, a) for _, a, b in selected]
    specs: list[LlmSpec] = []
    for pre, post in directions:
        for regime in ("iid", "batch_bursty"):
            specs.append(LlmSpec(
                f"llm_{len(specs) + 1:02d}_{pre}_to_{post}_{regime}",
                pre, post, regime,
            ))
    return specs


def sample_llm_paths(frame: pd.DataFrame, spec: LlmSpec, split: str, score: np.ndarray,
                     seed: int, paths: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((paths, HORIZON, BATCH_SIZE), dtype=np.int64)
    split_mask = frame["split"].eq(split).to_numpy()
    categories = frame["category"].astype(str).to_numpy()
    for path in range(paths):
        for batch in range(HORIZON):
            category = spec.pre_category if batch < SHIFT_BATCH else spec.post_category
            candidates = np.flatnonzero(split_mask & (categories == category))
            if not len(candidates):
                raise ValueError(f"no {split} examples for category={category}")
            if spec.regime == "batch_bursty":
                order = candidates[np.argsort(score[candidates], kind="stable")]
                bins = [x for x in np.array_split(order, 4) if len(x)]
                candidates = bins[int(rng.integers(0, len(bins)))]
            output[path, batch] = rng.choice(candidates, BATCH_SIZE, replace=True)
    return output


def run_llm(args: argparse.Namespace) -> pd.DataFrame:
    frame = load_llm_frame(args.cheap_scores, args.strong_scores)
    score = fit_llm_predictor(frame)
    embeddings = embedding_features(frame)
    contexts, pi, cheap_reward, strong_reward = fit_llm_contexts(frame, embeddings)
    robust = fit_robust_scores(frame, embeddings)
    gain_all = frame["gain"].to_numpy(dtype=np.float64)
    specs = make_llm_specs(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "llm_environment_manifest.json").write_text(
        json.dumps([asdict(x) for x in specs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    for position, spec in enumerate(specs, 1):
        print(f"[llm {position:02d}/40] {spec.environment_id}", flush=True)
        calibration_idx = sample_llm_paths(
            frame, spec, "calibration", score, 4_000_000 + position,
            args.calibration_paths,
        )
        calibration_score = score[calibration_idx]
        calibration_gain = gain_all[calibration_idx]
        calibration_artifacts = {
            "score_batch_dp": build_batch_dp(calibration_score, max(args.budgets)),
            "gain_batch_dp": build_batch_dp(calibration_gain, max(args.budgets)),
            "sequential_dp": build_sequential_dp(calibration_score, max(args.budgets)),
            "thresholds": {
                budget: threshold_from_calibration(calibration_score, budget)
                for budget in args.budgets
            },
        }
        for seed in range(args.seeds):
            idx = sample_llm_paths(
                frame, spec, "test", score, 5_000_000 + position * 1000 + seed, 1
            ).reshape(-1)
            evaluate_methods(
                gain=gain_all[idx], score=score[idx], robust_score=robust[idx],
                contexts=contexts[idx], pi=pi, cheap_reward=cheap_reward,
                strong_reward=strong_reward, calibration_score=calibration_score,
                calibration_gain=calibration_gain, budgets=args.budgets, seed=seed,
                environment_type="llm", environment_id=spec.environment_id, rows=rows,
                calibration_artifacts=calibration_artifacts,
            )
    result = pd.DataFrame(rows)
    result.to_parquet(args.output / "llm_metrics.parquet", index=False)
    result.drop(columns="batch_allocations").to_csv(
        args.output / "llm_metrics.csv", index=False
    )
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group = frame.groupby(
        ["environment_type", "environment_id", "budget", "budget_fraction", "method"],
        sort=True,
    )["total_realized_gain"]
    summary = group.agg(["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    return summary.rename(columns={"mean": "mean_total_realized_gain",
                                   "std": "std_total_realized_gain",
                                   "count": "num_seeds"})


def main() -> None:
    args = parse_args()
    if any(b < 0 or b > HORIZON * BATCH_SIZE for b in args.budgets):
        raise ValueError("budgets must be between 0 and 480")
    frames = []
    if args.part in ("synthetic", "all"):
        frames.append(run_synthetic(args))
    if args.part in ("llm", "all"):
        frames.append(run_llm(args))
    if args.part != "all":
        counterpart = args.output / (
            "llm_metrics.parquet" if args.part == "synthetic" else "synthetic_metrics.parquet"
        )
        if counterpart.exists():
            frames.append(pd.read_parquet(counterpart))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(args.output / "all_metrics.parquet", index=False)
    summary = summarize(combined)
    summary.to_csv(args.output / "all_summary.csv", index=False)
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
