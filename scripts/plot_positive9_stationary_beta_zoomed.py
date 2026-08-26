"""Plot the nine positive-margin stationary Beta environments in a 3x3 grid."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from plot_stationary_beta_suite import METHODS, STYLES
from plot_top12_stationary_beta import rank_environments


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "stationary_beta_suite"
OUTPUT = ROOT / "output" / "stationary_beta_positive9_zoomed"
NON_ORACLE_COMPETITORS = (
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)

mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def paired_delta_summary(metrics: pd.DataFrame, selected_ids: set[str]) -> pd.DataFrame:
    """Compute policy-minus-BatchPace paired differences over the same seeds."""
    selected = metrics.loc[metrics["environment_id"].isin(selected_ids)].copy()
    keys = ["environment_id", "budget", "seed"]
    batchpace = selected.loc[
        selected["method"].eq("BatchPace"), keys + ["total_realized_gain"]
    ].rename(columns={"total_realized_gain": "batchpace_gain"})
    competitors = selected.loc[
        selected["method"].isin(NON_ORACLE_COMPETITORS),
        keys + ["method", "total_realized_gain"],
    ]
    paired = competitors.merge(batchpace, on=keys, validate="many_to_one")
    paired["delta_vs_batchpace"] = (
        paired["total_realized_gain"] - paired["batchpace_gain"]
    )
    grouped = paired.groupby(["environment_id", "budget", "method"], sort=True)[
        "delta_vs_batchpace"
    ]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    return summary


def plot_total_gain(ax: plt.Axes, data: pd.DataFrame, collect_legend: bool):
    handles = []
    labels = []
    for method in METHODS:
        subset = data.loc[data["method"].eq(method)].sort_values("budget")
        style = STYLES[method]
        x = subset["budget"].to_numpy()
        y = subset["mean"].to_numpy()
        ci = subset["ci95"].to_numpy()
        linewidth = 1.90 if method == "BatchPace" else 1.05
        markersize = 3.7 if method == "BatchPace" else 2.7
        (line,) = ax.plot(
            x,
            y,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
            markersize=markersize,
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
        if collect_legend:
            handles.append(line)
            labels.append(method)

    lower = (data["mean"] - data["ci95"]).min()
    upper = (data["mean"] + data["ci95"]).max()
    span = max(float(upper - lower), 1.0)
    ax.set_ylim(min(0.0, float(lower)) - 0.04 * span, float(upper) + 0.07 * span)
    ax.set_xlim(-10, 202)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 192])
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=7.4, length=2.6, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
    ax.grid(True, alpha=0.20)
    return handles, labels


def plot_delta(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.axhline(
        0,
        color=STYLES["BatchPace"]["color"],
        linewidth=1.65,
        alpha=0.95,
        zorder=6,
    )
    for method in NON_ORACLE_COMPETITORS:
        subset = data.loc[data["method"].eq(method)].sort_values("budget")
        style = STYLES[method]
        x = subset["budget"].to_numpy()
        y = subset["mean"].to_numpy()
        ci = subset["ci95"].to_numpy()
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.00,
            markersize=2.6,
            alpha=0.92,
            zorder=4,
        )
        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=style["color"],
            alpha=0.09,
            linewidth=0,
            zorder=2,
        )

    lower = (data["mean"] - data["ci95"]).min()
    upper = (data["mean"] + data["ci95"]).max()
    lower = min(float(lower), 0.0)
    upper = max(float(upper), 0.0)
    span = max(upper - lower, 0.25)
    ax.set_ylim(lower - 0.10 * span, upper + 0.10 * span)
    ax.set_xlim(-10, 202)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 192])
    ax.tick_params(labelsize=7.1, labelbottom=True, length=2.5, pad=1.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, alpha=0.20)
    ax.set_ylabel("Policy − BP", fontsize=7.1, labelpad=1.8)


def make_figure(
    total_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 7.85))
    outer = fig.add_gridspec(
        3,
        3,
        left=0.060,
        right=0.995,
        bottom=0.068,
        top=0.840,
        wspace=0.20,
        hspace=0.31,
    )
    legend_handles = []
    legend_labels = []

    for panel_index, rank_row in enumerate(ranking.itertuples()):
        row, column = divmod(panel_index, 3)
        inner = outer[row, column].subgridspec(
            2, 1, height_ratios=(2.50, 1.22), hspace=0.09
        )
        total_ax = fig.add_subplot(inner[0])
        delta_ax = fig.add_subplot(inner[1])
        total_data = total_summary.loc[
            total_summary["environment_id"].eq(rank_row.environment_id)
        ]
        delta_data = delta_summary.loc[
            delta_summary["environment_id"].eq(rank_row.environment_id)
        ]
        handles, labels = plot_total_gain(
            total_ax, total_data, collect_legend=(panel_index == 0)
        )
        if panel_index == 0:
            legend_handles = handles
            legend_labels = labels
        plot_delta(delta_ax, delta_data)

        panel = chr(ord("a") + panel_index)
        total_ax.set_title(
            f"({panel}) Beta(α={rank_row.alpha:g}, β={rank_row.beta:g})\n"
            f"BatchPace advantage={rank_row.mean_margin:+.3f}",
            fontsize=8.2,
            linespacing=0.95,
            pad=4,
        )

    fig.suptitle(
        "Nine Positive-Margin Stationary Beta Environments",
        fontsize=12.3,
        y=0.990,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.960),
        ncol=3,
        fontsize=7.5,
        frameon=True,
        handlelength=1.8,
        columnspacing=1.0,
        borderpad=0.30,
    )
    fig.text(
        0.5,
        0.892,
        "Upper: total realized gain. Lower: competitor − BatchPace "
        "(negative values favor BatchPace; Oracle omitted below).",
        ha="center",
        fontsize=7.8,
    )
    fig.supxlabel("Global budget (B)", fontsize=9.2, y=0.022)
    fig.supylabel("Total realized gain (upper panels)", fontsize=9.2, x=0.008)
    return fig


def main() -> None:
    total_summary = pd.read_csv(RESULT / "summary.csv")
    metrics = pd.read_parquet(RESULT / "metrics.parquet")
    ranking = rank_environments(total_summary)
    ranking = ranking.loc[ranking["mean_margin"].gt(0)].reset_index(drop=True)
    if len(ranking) != 9:
        raise ValueError(f"Expected exactly 9 positive-margin environments, found {len(ranking)}")
    ranking["rank"] = np.arange(1, 10)
    selected_ids = set(ranking["environment_id"])
    selected_total = total_summary.loc[
        total_summary["environment_id"].isin(selected_ids)
    ].copy()
    delta_summary = paired_delta_summary(metrics, selected_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT / "selection_positive9.csv", index=False)
    (OUTPUT / "selection_positive9.json").write_text(
        json.dumps(ranking.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    selected_total.to_json(
        OUTPUT / "total_gain_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )
    delta_summary.to_json(
        OUTPUT / "paired_difference_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    figure = make_figure(selected_total, delta_summary, ranking)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT / f"stationary_beta_positive9_3x3_zoomed.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(figure)
    print(ranking[["rank", "alpha", "beta", "mean_margin"]].to_string(index=False))
    print(f"Created positive-margin 3x3 zoomed panel in {OUTPUT}")


if __name__ == "__main__":
    main()
