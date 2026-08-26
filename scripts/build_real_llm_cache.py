"""Merge MMLU-Pro rows, two model outputs, and query embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "real_llm" / "splits" / "mmlu_pro_main_1500.parquet"
CHEAP = ROOT / "results" / "real_llm" / "cache_qwen3_0.6b_5shot.jsonl"
STRONG = ROOT / "results" / "real_llm" / "cache_qwen3_4b_5shot.jsonl"
EMBEDDINGS = ROOT / "results" / "real_llm" / "qwen3_embedding_0.6b.jsonl"
OUTPUT_DIR = ROOT / "data" / "real_llm" / "cache"


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as stream:
        return pd.DataFrame(json.loads(line) for line in stream if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique(frame: pd.DataFrame, name: str) -> None:
    if len(frame) != 1500 or frame["question_id"].nunique() != 1500:
        raise AssertionError(f"{name} must contain 1500 unique question IDs")


def main() -> None:
    source = pd.read_parquet(SOURCE)
    cheap = read_jsonl(CHEAP)
    strong = read_jsonl(STRONG)
    embeddings = read_jsonl(EMBEDDINGS)
    for name, frame in (("source", source), ("cheap", cheap), ("strong", strong), ("embeddings", embeddings)):
        require_unique(frame, name)

    cheap_columns = {
        "prediction": "cheap_pred",
        "correct": "cheap_correct",
        "latency_seconds": "cheap_latency_seconds",
        "entropy": "cheap_entropy",
        "margin": "cheap_margin",
        "probabilities": "cheap_probabilities",
    }
    strong_columns = {
        "prediction": "strong_pred",
        "correct": "strong_correct",
        "latency_seconds": "strong_latency_seconds",
    }
    cheap = cheap[["question_id", *cheap_columns]].rename(columns=cheap_columns)
    strong = strong[["question_id", *strong_columns]].rename(columns=strong_columns)
    embeddings = embeddings[["question_id", "embedding"]]
    merged = source.merge(cheap, on="question_id", validate="one_to_one")
    merged = merged.merge(strong, on="question_id", validate="one_to_one")
    merged = merged.merge(embeddings, on="question_id", validate="one_to_one")
    for label in "ABCDEFGHIJ":
        merged[f"cheap_p_{label}"] = merged["cheap_probabilities"].map(lambda value: float(value[label]))
    merged["gain"] = merged["strong_correct"] - merged["cheap_correct"]
    merged["question_length"] = merged["question"].str.len()
    merged["num_options"] = merged["options"].map(len)
    merged = merged.drop(columns=["cheap_probabilities"])

    if merged["embedding"].map(len).nunique() != 1 or len(merged.iloc[0]["embedding"]) != 1024:
        raise AssertionError("Embeddings must all have dimension 1024")
    if not set(merged["gain"].unique()).issubset({-1, 0, 1}):
        raise AssertionError("Gain must be in {-1,0,1}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "mmlu_pro_qwen_pair_1500.parquet"
    merged.to_parquet(output, index=False)
    api = HfApi()
    model_files = {
        "Qwen3-0.6B-Q8_0": ROOT / "data" / "real_llm" / "models" / "qwen3-0.6b" / "Qwen3-0.6B-Q8_0.gguf",
        "Qwen3-4B-Q4_K_M": ROOT / "data" / "real_llm" / "models" / "qwen3-4b" / "Qwen3-4B-Q4_K_M.gguf",
        "Qwen3-Embedding-0.6B-Q8_0": ROOT / "data" / "real_llm" / "models" / "qwen3-embedding-0.6b" / "Qwen3-Embedding-0.6B-Q8_0.gguf",
    }
    manifest = {
        "rows": len(merged),
        "split_counts": merged["split"].value_counts().to_dict(),
        "cheap_accuracy": float(merged["cheap_correct"].mean()),
        "strong_accuracy": float(merged["strong_correct"].mean()),
        "gain_counts": {str(key): int(value) for key, value in merged["gain"].value_counts().sort_index().items()},
        "dataset_revision": api.repo_info("TIGER-Lab/MMLU-Pro", repo_type="dataset").sha,
        "model_revisions": {
            "Qwen/Qwen3-0.6B-GGUF": api.repo_info("Qwen/Qwen3-0.6B-GGUF").sha,
            "Qwen/Qwen3-4B-GGUF": api.repo_info("Qwen/Qwen3-4B-GGUF").sha,
            "Qwen/Qwen3-Embedding-0.6B-GGUF": api.repo_info("Qwen/Qwen3-Embedding-0.6B-GGUF").sha,
        },
        "model_sha256": {name: sha256(path) for name, path in model_files.items()},
        "output": str(output.resolve()),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
