"""Plot the 40 stationary single-subject LLM-derived environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "results/stationary_llm_suite"
DEFAULT_OUTPUT = ROOT / "output/stationary_llm_40"

# Keep the previously requested LLM presentation: no Oracle or Sequential Pacing.
METHODS = (
    "BatchPace",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)
STYLES = {
    "BatchPace": {
        "color": "#D62728", "marker": "o", "linewidth": 3.4,
        "markersize": 7.2, "alpha": 1.0, "zorder": 6,
    },
    "Fixed Threshold": {
        "color": "#009E73", "marker": "^", "linewidth": 1.9,
        "markersize": 5.2, "alpha": 0.82, "zorder": 3,
    },
    "Equal Split": {
        "color": "#9467BD", "marker": "D", "linewidth": 1.9,
        "markersize": 5.0, "alpha": 0.82, "zorder": 3,
    },
    "WISERouter-Offline": {
        "color": "#1F77B4", "marker": "P", "linewidth": 1.9,
        "markersize": 5.2, "alpha": 0.82, "zorder": 3,
    },
    "Robust Batch-Level Query Routing": {
        "color": "#E69F00", "marker": "X", "linewidth": 1.9,
        "markersize": 5.2, "alpha": 0.82, "zorder": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["method"].isin(METHODS)].copy()
    grouped = selected.groupby(
        ["environment_id", "budget", "method"], sort=True
    )["total_realized_gain"]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["ci95"] = 1.96 * result["std"].fillna(0.0) / np.sqrt(result["count"])
    return result


def make_figure(spec: dict[str, object], data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    for method in METHODS:
        subset = data.loc[data["method"].eq(method)].sort_values("budget")
        style = STYLES[method]
        x = subset["budget"].to_numpy()
        y = subset["mean"].to_numpy()
        ci = subset["ci95"].to_numpy()
        ax.plot(
            x,
            y,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=style["color"],
            alpha=0.17 if method == "BatchPace" else 0.07,
            linewidth=0,
            zorder=style["zorder"] - 1,
        )

    lower = float((data["mean"] - data["ci95"]).min())
    upper = float((data["mean"] + data["ci95"]).max())
    span = max(1.0, upper - lower)
    ax.set_ylim(min(0.0, lower) - 0.04 * span, max(0.0, upper) + 0.06 * span)
    ax.axhline(0, color="#777777", linewidth=0.9, zorder=0)
    ax.set_xticks(sorted(data["budget"].unique()))
    ax.set_xlabel("Global budget (B)", fontsize=17)
    ax.set_ylabel("Total realized gain", fontsize=17)
    number = str(spec["environment_id"]).split("_")[2]
    ax.set_title(
        f"Stationary LLM-derived environment {number}: {spec['subject_label']}\n"
        f"Category: {str(spec['category']).title()} | 24 IID batches | One fixed subject pool",
        fontsize=17,
        pad=14,
    )
    ax.grid(True, alpha=0.20)
    ax.tick_params(labelsize=13)
    ax.legend(loc="best", fontsize=12, frameon=True, ncol=2)
    return fig


def main() -> None:
    args = parse_args()
    metrics = pd.read_parquet(args.result_dir / "metrics.parquet")
    specs = json.loads((args.result_dir / "environment_manifest.json").read_text(encoding="utf-8"))
    if len(specs) != 40:
        raise AssertionError(f"expected 40 environment specs, found {len(specs)}")
    summary = summarize(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "figure_data.csv", index=False)
    summary.to_json(
        args.output_dir / "figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    atlas_path = args.output_dir / "stationary_llm_40_atlas.pdf"
    manifest_rows = []
    with PdfPages(atlas_path) as atlas:
        for spec in specs:
            environment_id = str(spec["environment_id"])
            data = summary.loc[summary["environment_id"].eq(environment_id)]
            if data.empty:
                raise ValueError(f"missing metrics for {environment_id}")
            figure = make_figure(spec, data)
            png_path = args.output_dir / f"{environment_id}.png"
            svg_path = args.output_dir / f"{environment_id}.svg"
            figure.savefig(png_path, dpi=240, facecolor="white")
            figure.savefig(svg_path, facecolor="white")
            atlas.savefig(figure, facecolor="white")
            plt.close(figure)
            manifest_rows.append({
                "environment_id": environment_id,
                "category": spec["category"],
                "source": spec["source"],
                "subject_label": spec["subject_label"],
                "png": str(png_path.relative_to(ROOT)),
                "svg": str(svg_path.relative_to(ROOT)),
                "num_seeds": int(data["count"].min()),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "figure_manifest.csv", index=False)
    if len(manifest) != 40:
        raise AssertionError(f"expected 40 figures, created {len(manifest)}")
    print(f"created 40 PNG + 40 SVG figures and {atlas_path}", flush=True)


if __name__ == "__main__":
    main()

