"""Print compact diagnostics for one or more restartable inference caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as stream:
        return pd.DataFrame(json.loads(line) for line in stream if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        frame = read_jsonl(path)
        print(f"\n{path}")
        print(f"rows={len(frame)} accuracy={frame['correct'].mean():.4f}")
        print(
            f"latency_mean={frame['latency_seconds'].mean():.3f}s "
            f"latency_p95={frame['latency_seconds'].quantile(.95):.3f}s"
        )
        print("predictions")
        print(frame["prediction"].fillna("<none>").value_counts().sort_index().to_string())
        print("accuracy_by_split")
        print(frame.groupby("split")["correct"].agg(["count", "mean"]).to_string())
        print("accuracy_by_category")
        print(frame.groupby("category")["correct"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
