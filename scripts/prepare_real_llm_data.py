"""Create a fixed, stratified MMLU-Pro split for the real-LLM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 20260825


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/real_llm/mmlu_pro/data/test-00000-of-00001.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_llm/splits"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.source)

    selected, _ = train_test_split(
        source,
        train_size=1500,
        random_state=args.seed,
        stratify=source["category"],
    )
    train, remainder = train_test_split(
        selected,
        train_size=500,
        random_state=args.seed + 1,
        stratify=selected["category"],
    )
    calibration, test = train_test_split(
        remainder,
        train_size=500,
        random_state=args.seed + 2,
        stratify=remainder["category"],
    )

    frames = []
    for split_name, frame in (
        ("train", train),
        ("calibration", calibration),
        ("test", test),
    ):
        frame = frame.copy()
        frame["split"] = split_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["split", "category", "question_id"]).reset_index(drop=True)
    combined.to_parquet(args.output_dir / "mmlu_pro_main_1500.parquet", index=False)

    # A balanced-enough smoke test: one question from every category, then fill to 20.
    pilot_parts = [group.sample(1, random_state=args.seed) for _, group in combined.groupby("category")]
    pilot = pd.concat(pilot_parts)
    remaining = combined.drop(index=pilot.index)
    pilot = pd.concat(
        [pilot, remaining.sample(20 - len(pilot), random_state=args.seed + 3)]
    ).sample(frac=1, random_state=args.seed + 4)
    pilot.to_parquet(args.output_dir / "mmlu_pro_pilot_20.parquet", index=False)

    api = HfApi()
    manifest = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": api.repo_info("TIGER-Lab/MMLU-Pro", repo_type="dataset").sha,
        "source_file": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "seed": args.seed,
        "rows": {name: int((combined["split"] == name).sum()) for name in ["train", "calibration", "test"]},
        "categories": {
            split: combined.loc[combined["split"] == split, "category"].value_counts().sort_index().to_dict()
            for split in ["train", "calibration", "test"]
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
