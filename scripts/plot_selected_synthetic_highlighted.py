"""Create a balanced 3x4 synthetic panel with BatchPace emphasized."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "benchmark_suite" / "synthetic_metrics.parquet"
MANIFEST = ROOT / "results" / "benchmark_suite" / "synthetic_environment_manifest.json"
OUTPUT = ROOT / "output" / "selected_synthetic_highlighted"

# Balanced selection: every shift appears three times, both regimes appear six
# times, and all five gain families are represented.
SELECTED_NUMBERS = (1, 4, 13, 16, 17, 20, 23, 26, 29, 32, 35, 38)

# Sequential Pacing is intentionally excluded; Oracle BatchPace is retained.
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
        "linewidth": 4.4,
        "markersize": 9.0,
        "alpha": 1.0,
        "zorder": 7,
    },
    "Oracle BatchPace": {
        "color": "#333333",
        "marker": "*",
        "linestyle": "--",
        "linewidth": 2.8,
        "markersize": 9.0,
        "alpha": 0.82,
        "zorder": 5,
    },
    "Fixed Threshold": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 7.2,
        "alpha": 0.82,
        "zorder": 3,
    },
    "Equal Split": {
        "color": "#9467BD",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 7.0,
        "alpha": 0.82,
        "zorder": 3,
    },
    "WISERouter-Offline": {
        "color": "#1F77B4",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 7.2,
        "alpha": 0.82,
        "zorder": 3,
    },
    "Robust Batch-Level Query Routing": {
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 7.2,
        "alpha": 0.82,
        "zorder": 3,
    },
}

FAMILY_LABELS = {
    "sparse": ("Sparse", "0.25·Beta(2,5) + 0.75·δ₀"),
    "moderate": ("Moderate", "Beta(2.5,4)"),
    "bimodal": ("Bimodal", "0.35·Beta(6,2) + 0.65·Beta(1.5,7)"),
    "heavy_tail": ("Heavy-tail", "Beta(0.75,3.5)"),
    "rare_jackpot": (
        "Rare-jackpot",
        "0.08·Beta(12,1.5) + 0.92·Beta(0.5,12)",
    ),
}

SHIFT_LABELS = {
    "mean_up": "Mean-up",
    "mean_down": "Mean-down",
    "variance_up": "Variance-up",
    "tail_reversal": "Tail-reversal",
}


def load_selected_manifest() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_number = {int(row["environment_id"][4:6]): row for row in manifest}
    selected = [by_number[number] for number in SELECTED_NUMBERS]
    if len(selected) != 12:
        raise ValueError("Expected exactly 12 selected synthetic environments")
    return selected


def summarize(metrics: pd.DataFrame, environment_ids: list[str]) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["environment_id"].isin(environment_ids)
        & metrics["method"].isin(METHODS)
    ].copy()
    grouped = selected.groupby(
        ["environment_id", "budget", "method"], sort=True
    )["total_realized_gain"]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["ci95"] = 1.96 * result["std"].fillna(0) / np.sqrt(result["count"])
    return result


def panel_title(row: dict, panel_index: int) -> str:
    family, beta = FAMILY_LABELS[row["family"]]
    shift = SHIFT_LABELS[row["shift"]]
    regime = "IID" if row["regime"] == "iid" else "Batch-bursty"
    panel = chr(ord("a") + panel_index)
    return f"({panel}) {family}: {beta}\n{shift} | {regime}"


def make_grid_figure(summary: pd.DataFrame, selected: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(30.0, 21.0),
        sharex=False,
        sharey=False,
    )

    handles = []
    labels = []
    for panel_index, (ax, environment) in enumerate(zip(axes.flat, selected)):
        data = summary.loc[
            summary["environment_id"].eq(environment["environment_id"])
        ]
        for method in METHODS:
            subset = data.loc[data["method"].eq(method)].sort_values("budget")
            style = STYLES[method]
            x = subset["budget"].to_numpy()
            y = subset["mean"].to_numpy()
            ci = subset["ci95"].to_numpy()
            (line,) = ax.plot(
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
            if panel_index == 0:
                handles.append(line)
                labels.append(method)

        ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        lower = (data["mean"] - data["ci95"]).min()
        upper = (data["mean"] + data["ci95"]).max()
        span = max(upper - lower, 1.0)
        ax.set_ylim(min(0.0, lower) - 0.04 * span, upper + 0.07 * span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        ax.set_title(panel_title(environment, panel_index), fontsize=16.5, pad=13)
        ax.grid(True, alpha=0.20)
        ax.tick_params(labelsize=14, labelbottom=True)
        ax.set_xticks([0, 48, 96, 144, 192])
        ax.set_xlim(-10, 202)

    fig.suptitle(
        "Selected Synthetic Environments: Budget–Gain Comparison",
        fontsize=24,
        y=0.992,
    )
    fig.supxlabel("Global budget (B)", fontsize=21, y=0.020)
    fig.supylabel("Total realized gain", fontsize=21, x=0.006)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.960),
        ncol=6,
        fontsize=16,
        frameon=True,
    )
    fig.text(
        0.5,
        0.918,
        "Panel-specific y-axis ranges are used to enlarge performance differences",
        ha="center",
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.058,
        right=0.994,
        bottom=0.065,
        top=0.865,
        wspace=0.18,
        hspace=0.34,
    )
    return fig


def main() -> None:
    selected = load_selected_manifest()
    environment_ids = [row["environment_id"] for row in selected]
    metrics = pd.read_parquet(METRICS)
    summary = summarize(metrics, environment_ids)

    expected_rows = 12 * len(METHODS) * metrics["budget"].nunique()
    if len(summary) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summary rows, found {len(summary)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_json(
        OUTPUT / "selected_synthetic_figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )
    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure = make_grid_figure(summary, selected)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT / f"selected_synthetic_3x4_zoomed_xlabels.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(figure)
    print(f"Created zoomed 3x4 synthetic panel in {OUTPUT}")


if __name__ == "__main__":
    main()
