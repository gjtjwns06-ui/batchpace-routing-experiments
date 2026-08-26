"""Plot one highlighted budget-gain curve for each stationary Beta environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "results" / "stationary_beta_suite"
DEFAULT_OUTPUT = ROOT / "output" / "stationary_beta_40"

METHODS = (
    "BatchPace",
    "Oracle BatchPace",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)

STYLES = {
    "BatchPace": {
        "color": "#D62728",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3.6,
        "markersize": 7.5,
        "alpha": 1.0,
        "zorder": 7,
    },
    "Oracle BatchPace": {
        "color": "#333333",
        "marker": "*",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 8.0,
        "alpha": 0.84,
        "zorder": 5,
    },
    "Fixed Threshold": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 5.5,
        "alpha": 0.84,
        "zorder": 3,
    },
    "Equal Split": {
        "color": "#9467BD",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 5.2,
        "alpha": 0.84,
        "zorder": 3,
    },
    "WISERouter-Offline": {
        "color": "#1F77B4",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 5.5,
        "alpha": 0.84,
        "zorder": 3,
    },
    "Robust Batch-Level Query Routing": {
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 5.5,
        "alpha": 0.84,
        "zorder": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_figure(data: pd.DataFrame) -> plt.Figure:
    row = data.iloc[0]
    number = int(row["environment_number"])
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    theoretical_mean = float(row["theoretical_mean"])

    fig, ax = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)
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
            linestyle=style["linestyle"],
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
            alpha=0.16 if method == "BatchPace" else 0.055,
            linewidth=0,
            zorder=style["zorder"] - 1,
        )

    lower = (data["mean"] - data["ci95"]).min()
    upper = (data["mean"] + data["ci95"]).max()
    span = max(float(upper - lower), 1.0)
    ax.set_ylim(min(0.0, float(lower)) - 0.04 * span, float(upper) + 0.07 * span)
    ax.set_xlim(-10, 202)
    budgets = sorted(int(value) for value in data["budget"].unique())
    ax.set_xticks(budgets)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
    ax.grid(True, alpha=0.20)
    ax.tick_params(labelsize=12.5)
    ax.set_xlabel("Global budget (B)", fontsize=15)
    ax.set_ylabel("Total realized gain", fontsize=15)
    ax.set_title(
        f"Stationary synthetic environment {number:02d}: "
        f"Beta(α={alpha:g}, β={beta:g})\n"
        f"24 identical-distribution batches | E[gain]={theoretical_mean:.3f}",
        fontsize=16,
        pad=12,
    )
    ax.legend(loc="best", fontsize=10.5, frameon=True, ncol=2)
    return fig


def main() -> None:
    args = parse_args()
    summary_path = args.result_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    if summary["environment_id"].nunique() != 40:
        raise ValueError("Expected exactly 40 environments")
    if set(summary["method"].unique()) != set(METHODS):
        raise ValueError("Unexpected method set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_json(
        args.output_dir / "figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    manifest_rows = []
    atlas_path = args.output_dir / "stationary_beta_40_atlas.pdf"
    with PdfPages(atlas_path) as pdf:
        for environment_id, data in summary.groupby("environment_id", sort=False):
            data = data.sort_values(["budget", "method"])
            number = int(data["environment_number"].iloc[0])
            alpha = float(data["alpha"].iloc[0])
            beta = float(data["beta"].iloc[0])
            fig = make_figure(data)
            stem = f"beta_{number:02d}_alpha_{alpha:g}_beta_{beta:g}".replace(".", "p")
            png_path = args.output_dir / f"{stem}.png"
            svg_path = args.output_dir / f"{stem}.svg"
            fig.savefig(png_path, dpi=220)
            fig.savefig(svg_path)
            pdf.savefig(fig)
            plt.close(fig)
            manifest_rows.append(
                {
                    "environment_number": number,
                    "environment_id": environment_id,
                    "alpha": alpha,
                    "beta": beta,
                    "png": str(png_path.relative_to(ROOT)),
                    "svg": str(svg_path.relative_to(ROOT)),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values("environment_number")
    manifest.to_csv(args.output_dir / "figure_manifest.csv", index=False)
    print(f"Created {len(manifest)} PNG and {len(manifest)} SVG figures")
    print(f"PDF atlas: {atlas_path}")


if __name__ == "__main__":
    main()
