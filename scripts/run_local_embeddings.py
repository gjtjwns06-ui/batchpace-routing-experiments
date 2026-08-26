"""Create restartable Qwen query embeddings through llama.cpp."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def embedding_text(row: pd.Series) -> str:
    options = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(row["options"]))
    return (
        "Instruct: Represent this multiple-choice question for semantic clustering.\n"
        f"Query: {row['question']}\nOptions:\n{options}"
    )


def completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as stream:
        return {int(json.loads(line)["question_id"]) for line in stream if line.strip()}


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input).sort_values("question_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(args.output)
    pending = frame.loc[~frame["question_id"].astype(int).isin(done)].reset_index(drop=True)
    print(f"rows={len(frame)} completed={len(done)} pending={len(pending)}", flush=True)

    with args.output.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start : start + args.batch_size]
            texts = [embedding_text(row) for _, row in batch.iterrows()]
            before = time.perf_counter()
            response = requests.post(
                f"{args.base_url.rstrip('/')}/v1/embeddings",
                json={"model": "local", "input": texts, "encoding_format": "float"},
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            elapsed = time.perf_counter() - before
            vectors = sorted(payload["data"], key=lambda item: item["index"])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding response size mismatch")
            for (_, row), item in zip(batch.iterrows(), vectors, strict=True):
                record = {
                    "question_id": int(row["question_id"]),
                    "split": str(row["split"]),
                    "category": str(row["category"]),
                    "embedding": item["embedding"],
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            completed = min(start + len(batch), len(pending))
            if start == 0 or completed % 100 == 0 or completed == len(pending):
                print(
                    f"{completed}/{len(pending)} batch_latency={elapsed:.2f}s "
                    f"dim={len(vectors[0]['embedding'])}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
