"""Run deterministic MMLU-Pro inference against a local llama.cpp server.

The script is restartable: completed question IDs are read from the JSONL cache and
skipped. Each request generates only the answer-label token with thinking disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


LABELS = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--few-shot-source", type=Path, default=None)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def format_question(row: pd.Series, include_answer: bool) -> str:
    options = "\n".join(f"{LABELS[i]}. {text}" for i, text in enumerate(row["options"]))
    answer = str(row["answer"]).strip().upper() if include_answer else ""
    return f"Question: {row['question']}\nOptions:\n{options}\nAnswer: {answer}"


def build_prompt(row: pd.Series, demonstrations: dict[str, list[pd.Series]]) -> str:
    parts = [
        "Choose the correct answer to each multiple-choice question. "
        "Follow the examples and reply to the final question with exactly one letter "
        "from A to J and no explanation."
    ]
    for example in demonstrations.get(str(row["category"]), []):
        parts.append(format_question(example, include_answer=True))
    parts.append(format_question(row, include_answer=False))
    return "\n\n".join(parts)


def load_demonstrations(path: Path | None, shots: int) -> dict[str, list[pd.Series]]:
    if path is None or shots <= 0:
        return {}
    frame = pd.read_parquet(path)
    result: dict[str, list[pd.Series]] = {}
    for category, group in frame.groupby("category", sort=True):
        result[str(category)] = [row for _, row in group.head(shots).iterrows()]
    return result


def normalize_label(text: str) -> str | None:
    match = re.search(r"\b([A-J])\b", text.upper())
    if match:
        return match.group(1)
    return None


def strict_label_token(text: str) -> str | None:
    match = re.fullmatch(r"\s*[\(\[\{]?([A-J])[\.\)\]\}:]?\s*", text.upper())
    return match.group(1) if match else None


def label_probabilities(choice: dict, prediction: str | None) -> dict[str, float]:
    scores = {label: 0.0 for label in LABELS}
    content = choice.get("logprobs", {}).get("content") or []
    selected = None
    for position in content:
        generated_label = strict_label_token(str(position.get("token", "")))
        candidate_labels = {
            strict_label_token(str(candidate.get("token", "")))
            for candidate in position.get("top_logprobs") or []
        }
        if generated_label or len(candidate_labels - {None}) >= 2:
            selected = position
            break
    if selected:
        candidates = [selected] + list(selected.get("top_logprobs") or [])
        for candidate in candidates:
            label = strict_label_token(str(candidate.get("token", "")))
            logprob = candidate.get("logprob")
            if label and isinstance(logprob, (int, float)):
                scores[label] = max(scores[label], math.exp(float(logprob)))
    total = sum(scores.values())
    if total <= 0 and prediction:
        scores[prediction] = 1.0
        total = 1.0
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


def completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                result.add(int(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return result


def request_answer(base_url: str, prompt: str, timeout: float) -> tuple[dict, float]:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 4,
        "logprobs": True,
        "top_logprobs": 50,
    }
    last_error: Exception | None = None
    for attempt in range(4):
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
    raise RuntimeError(f"Inference request failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    # Group categories so llama.cpp can reuse the shared few-shot prefix.
    frame = frame.sort_values(["category", "question_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    demonstrations = load_demonstrations(args.few_shot_source, args.shots)
    done = completed_ids(args.output)
    pending = frame.loc[~frame["question_id"].astype(int).isin(done)]
    print(f"rows={len(frame)} completed={len(done)} pending={len(pending)}", flush=True)

    with args.output.open("a", encoding="utf-8") as sink:
        for position, (_, row) in enumerate(pending.iterrows(), start=1):
            prompt = build_prompt(row, demonstrations)
            response, latency = request_answer(args.base_url, prompt, args.timeout)
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content") or choice.get("text") or ""
            prediction = normalize_label(str(content))
            probabilities = label_probabilities(choice, prediction)
            if prediction is None and sum(probabilities.values()) > 0:
                prediction = max(probabilities, key=probabilities.get)
            gold = str(row["answer"]).strip().upper()
            record = {
                "question_id": int(row["question_id"]),
                "split": str(row.get("split", "unknown")),
                "category": str(row["category"]),
                "model_id": args.model_id,
                "prediction": prediction,
                "raw_content": str(content),
                "gold": gold,
                "correct": int(prediction == gold),
                "latency_seconds": latency,
                "prompt_chars": len(prompt),
                "option_count": int(len(row["options"])),
                "probabilities": probabilities,
                "entropy": float(
                    -sum(p * math.log(max(p, 1e-12)) for p in probabilities.values() if p > 0)
                ),
                "margin": float(
                    np.diff(sorted(probabilities.values(), reverse=True)[:2])[0] * -1
                    if len(probabilities) >= 2
                    else 1.0
                ),
                "usage": response.get("usage", {}),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            if position == 1 or position % 10 == 0 or position == len(pending):
                print(
                    f"{position}/{len(pending)} id={record['question_id']} pred={prediction} "
                    f"gold={gold} correct={record['correct']} latency={latency:.2f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
