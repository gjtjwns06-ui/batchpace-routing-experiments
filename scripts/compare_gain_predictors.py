"""Select a gain predictor by train-only out-of-fold ranking quality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "real_llm" / "cache" / "mmlu_pro_qwen_pair_1500.parquet"
OUTPUT = ROOT / "results" / "real_llm" / "gain_predictor_comparison.json"


def main() -> None:
    frame = pd.read_parquet(CACHE)
    train = frame["split"].eq("train").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    embeddings = np.vstack(frame["embedding"].map(np.asarray))
    pca = PCA(n_components=64, random_state=0).fit(embeddings[train])
    reduced = normalize(pca.transform(embeddings))
    category = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    category_train = category.fit_transform(frame.loc[train, ["category"]])
    category_all = category.transform(frame[["category"]])
    numeric_columns = [f"cheap_p_{label}" for label in "ABCDEFGHIJ"] + [
        "cheap_entropy", "cheap_margin", "question_length", "num_options"
    ]
    scaler = StandardScaler().fit(frame.loc[train, numeric_columns])
    numeric = scaler.transform(frame[numeric_columns])
    x = np.column_stack([numeric, category_all, reduced])
    y = frame["gain"].to_numpy(dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260825)
    models = {
        "ridge-rich-a1": Ridge(alpha=1.0),
        "ridge-rich-a10": Ridge(alpha=10.0),
        "ridge-rich-a100": Ridge(alpha=100.0),
        "hist-gradient-boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=2.0, random_state=0
        ),
        "random-forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=8, max_features=0.7,
            n_jobs=-1, random_state=0
        ),
        "extra-trees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=8, max_features=0.7,
            n_jobs=-1, random_state=0
        ),
    }
    rows = []
    oof_predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        oof = cross_val_predict(model, x[train], y[train], cv=cv, n_jobs=1)
        oof_predictions[name] = oof
        order = np.argsort(-oof, kind="stable")
        top_gains = {f"top_{k}_gain": float(y[train][order[:k]].sum()) for k in (25, 50, 100)}
        rows.append({
            "model": name,
            "cv_spearman": float(spearmanr(oof, y[train]).statistic),
            "cv_mae": float(mean_absolute_error(y[train], oof)),
            **top_gains,
        })
        print(rows[-1], flush=True)

    heuristic_scores = {
        "heuristic-one-minus-maxprob": 1.0 - frame.loc[train, [f"cheap_p_{label}" for label in "ABCDEFGHIJ"]].max(axis=1).to_numpy(),
        "heuristic-entropy": frame.loc[train, "cheap_entropy"].to_numpy(),
        "heuristic-negative-margin": -frame.loc[train, "cheap_margin"].to_numpy(),
    }
    for name, oof in heuristic_scores.items():
        order = np.argsort(-oof, kind="stable")
        rows.append({
            "model": name,
            "cv_spearman": float(spearmanr(oof, y[train]).statistic),
            "cv_mae": float("nan"),
            **{f"top_{k}_gain": float(y[train][order[:k]].sum()) for k in (25, 50, 100)},
        })
        print(rows[-1], flush=True)

    ranked = sorted(rows, key=lambda row: (row["top_50_gain"], row["cv_spearman"]), reverse=True)
    selected_name = ranked[0]["model"]
    if selected_name == "heuristic-one-minus-maxprob":
        test_prediction = 1.0 - frame.loc[test, [f"cheap_p_{label}" for label in "ABCDEFGHIJ"]].max(axis=1).to_numpy()
    elif selected_name == "heuristic-entropy":
        test_prediction = frame.loc[test, "cheap_entropy"].to_numpy()
    elif selected_name == "heuristic-negative-margin":
        test_prediction = -frame.loc[test, "cheap_margin"].to_numpy()
    else:
        selected = models[selected_name]
        selected.fit(x[train], y[train])
        test_prediction = selected.predict(x[test])
    test_order = np.argsort(-test_prediction, kind="stable")
    test_metrics = {
        "spearman": float(spearmanr(test_prediction, y[test]).statistic),
        "mae": float(mean_absolute_error(y[test], test_prediction)),
        **{f"top_{k}_gain": float(y[test][test_order[:k]].sum()) for k in (25, 50, 100)},
    }
    result = {
        "selection_rule": "maximize train OOF top-50 gain, then OOF Spearman",
        "selected_model": selected_name,
        "train_oof": ranked,
        "test_diagnostic_after_selection": test_metrics,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
