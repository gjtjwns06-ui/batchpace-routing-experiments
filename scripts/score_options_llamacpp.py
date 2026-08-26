"""Score every multiple-choice option with a local llama.cpp server.

This is deliberately not free generation.  The server grammar restricts the
next token to the option labels, and ``post_sampling_probs`` returns the full
conditional distribution over that candidate set.  The cache is append-only
and restartable by question_id.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


LABELS = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--demonstrations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8088")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def format_question(row: pd.Series, answer: bool) -> str:
    labels = LABELS[: len(row["options"])]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, row["options"]))
    suffix = str(row["answer"]).strip().upper() if answer else ""
    return f"Question: {row['question']}\nOptions:\n{options}\nAnswer: {suffix}"


def demonstrations_by_category(path: Path, shots: int) -> dict[str, list[pd.Series]]:
    frame = pd.read_parquet(path)
    return {
        str(category): [row for _, row in group.head(shots).iterrows()]
        for category, group in frame.groupby("category", sort=True)
    }


def build_prompt(row: pd.Series, examples: dict[str, list[pd.Series]]) -> str:
    parts = [
        "Choose the correct answer to each multiple-choice question. Follow the "
        "examples. For the final question, select exactly one option label."
    ]
    parts.extend(format_question(x, True) for x in examples.get(str(row["category"]), []))
    parts.append(format_question(row, False))
    return "\n\n".join(parts)


def completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                result.add(int(json.loads(line)["question_id"]))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                pass
    return result


def compact_cache(path: Path) -> None:
    """Atomically remove duplicate IDs left by an interrupted concurrent run."""
    records: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            records[int(json.loads(line)["question_id"])] = line
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(records.values()) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_scores(base_url: str, prompt: str, labels: str, timeout: float) -> tuple[dict, float]:
    # temperature=1 preserves the model softmax; all truncation samplers are disabled.
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "seed": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": len(labels),
        "grammar": f"root ::= [{labels}]",
        "post_sampling_probs": True,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "min_keep": len(labels),
        "repeat_penalty": 1.0,
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            started = time.perf_counter()
            response = requests.post(
                f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout
            )
            latency = time.perf_counter() - started
            response.raise_for_status()
            return response.json(), latency
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"option scoring failed: {last_error}")


def parse_distribution(response: dict, labels: str) -> dict[str, float]:
    positions = response["choices"][0].get("logprobs", {}).get("content", [])
    if not positions:
        raise ValueError("server returned no token probabilities")
    candidates = positions[0].get("top_probs", [])
    probabilities = {label: 0.0 for label in labels}
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip()
        if token in probabilities:
            probabilities[token] = max(probabilities[token], float(candidate["prob"]))
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError(f"incomplete constrained distribution: {probabilities}")
    # llama.cpp can omit candidates whose post-grammar softmax underflows to zero.
    # Preserve them as the smallest positive float so their log-likelihood is
    # finite without changing the distribution at normal precision.
    floor = np.finfo(float).tiny
    probabilities = {label: max(value, floor) for label, value in probabilities.items()}
    total = sum(probabilities.values())
    return {label: value / total for label, value in probabilities.items()}


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    frame = frame.sort_values(["category", "question_id"]).reset_index(drop=True)
    examples = demonstrations_by_category(args.demonstrations, args.shots)
    done = completed_ids(args.output)
    pending = frame.loc[~frame["question_id"].astype(int).isin(done)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"rows={len(frame)} completed={len(done)} pending={len(pending)}", flush=True)
    pending_rows = [row for _, row in pending.iterrows()]

    def score_row(row: pd.Series) -> dict:
        labels = LABELS[: len(row["options"])]
        response, latency = request_scores(
            args.base_url, build_prompt(row, examples), labels, args.timeout
        )
        probability = parse_distribution(response, labels)
        log_likelihood = {
            label: float(math.log(max(value, np.finfo(float).tiny)))
            for label, value in probability.items()
        }
        prediction = max(labels, key=log_likelihood.get)
        gold = str(row["answer"]).strip().upper()
        values = np.asarray(list(probability.values()), dtype=np.float64)
        ordered = np.sort(values)[::-1]
        return {
            "question_id": int(row["question_id"]),
            "split": str(row["split"]),
            "category": str(row["category"]),
            "model_id": args.model_id,
            "scoring": "constrained_option_label_conditional_log_likelihood",
            "option_probability": probability,
            "option_log_likelihood": log_likelihood,
            "prediction": prediction,
            "gold": gold,
            "correct": int(prediction == gold),
            "entropy": float(-(values * np.log(np.maximum(values, 1e-300))).sum()),
            "margin": float(ordered[0] - ordered[1]),
            "latency_seconds": latency,
            "usage": response.get("usage", {}),
        }

    with args.output.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for position, record in enumerate(executor.map(score_row, pending_rows), 1):
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if position == 1 or position % 25 == 0 or position == len(pending):
                    print(
                        f"{position}/{len(pending)} id={record['question_id']} "
                        f"pred={record['prediction']} gold={record['gold']} "
                        f"latency={record['latency_seconds']:.3f}s",
                        flush=True,
                    )
    compact_cache(args.output)


if __name__ == "__main__":
    main()
