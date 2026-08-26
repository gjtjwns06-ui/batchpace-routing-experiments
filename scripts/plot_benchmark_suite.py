"""Create one budget-gain figure per environment and a combined PDF atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/benchmark_suite"
OUTPUT = ROOT / "output/benchmark_suite"

METHODS = (
    "BatchPace",
    "Oracle BatchPace",
    "Sequential Pacing",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)
STYLES = {
    "BatchPace": ("#0072B2", "o", "-"),
    "Oracle BatchPace": ("#000000", "*", "--"),
    "Sequential Pacing": ("#E69F00", "s", "-"),
    "Fixed Threshold": ("#009E73", "^", "-"),
    "Equal Split": ("#CC79A7", "D", "-"),
    "WISERouter-Offline": ("#56B4E9", "P", "-"),
    "Robust Batch-Level Query Routing": ("#D55E00", "X", "-"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, default=RESULT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    return p.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    frames = []
    for name in ("synthetic_metrics.parquet", "llm_metrics.parquet"):
        candidate = path / name
        if candidate.exists():
            frames.append(pd.read_parquet(candidate))
    if len(frames) != 2:
        raise FileNotFoundError("both synthetic_metrics.parquet and llm_metrics.parquet are required")
    return pd.concat(frames, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(
        ["environment_type", "environment_id", "budget", "method"], sort=True
    )["total_realized_gain"]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["ci95"] = 1.96 * result["std"].fillna(0) / np.sqrt(result["count"])
    return result


def make_figure(environment_type: str, environment_id: str,
                data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for method in METHODS:
        subset = data.loc[data["method"].eq(method)].sort_values("budget")
        color, marker, line = STYLES[method]
        x = subset["budget"].to_numpy()
        y = subset["mean"].to_numpy()
        ci = subset["ci95"].to_numpy()
        ax.plot(x, y, label=method, color=color, marker=marker, linestyle=line,
                linewidth=1.8, markersize=4.8)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.10, linewidth=0)
    ax.axhline(0, color="#777777", linewidth=0.7, zorder=0)
    ax.set_xlabel("Global budget (B)")
    ax.set_ylabel("Total realized gain")
    kind = "Synthetic" if environment_type == "synthetic" else "LLM-derived"
    ax.set_title(f"{kind} environment: {environment_id}", fontsize=10.5)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=7.0, frameon=True, ncol=2)
    return fig


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.result_dir)
    summary = summarize(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "figure_summary.csv", index=False)
    atlas = ROOT / "output/pdf/batchpace_80_environment_atlas.pdf"
    atlas.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    with PdfPages(atlas) as pdf:
        for (kind, environment), group in summary.groupby(
            ["environment_type", "environment_id"], sort=True
        ):
            directory = args.output_dir / str(kind)
            directory.mkdir(parents=True, exist_ok=True)
            fig = make_figure(str(kind), str(environment), group)
            png = directory / f"{environment}.png"
            svg = directory / f"{environment}.svg"
            fig.savefig(png, dpi=180, facecolor="white")
            fig.savefig(svg, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            manifest_rows.append({
                "environment_type": kind,
                "environment_id": environment,
                "png": str(png.relative_to(ROOT)),
                "svg": str(svg.relative_to(ROOT)),
                "seeds": int(group["count"].min()),
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "figure_manifest.csv", index=False)
    if len(manifest) != 80:
        raise AssertionError(f"expected 80 figures, got {len(manifest)}")
    print(f"created {len(manifest)} PNG + {len(manifest)} SVG figures", flush=True)
    print(f"atlas={atlas}", flush=True)


if __name__ == "__main__":
    main()
