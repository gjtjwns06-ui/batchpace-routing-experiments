"""Select and plot the 12 stationary Beta environments best for BatchPace."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from plot_stationary_beta_suite import METHODS, STYLES


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "stationary_beta_suite" / "summary.csv"
OUTPUT = ROOT / "output" / "stationary_beta_top12"


def rank_environments(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank by mean advantage over the strongest non-Oracle baseline at B>0."""
    nonzero = summary.loc[summary["budget"].gt(0)].copy()
    batchpace = nonzero.loc[
        nonzero["method"].eq("BatchPace"),
        ["environment_id", "environment_number", "alpha", "beta", "budget", "mean"],
    ].rename(columns={"mean": "batchpace_gain"})
    competitor = (
        nonzero.loc[
            ~nonzero["method"].isin(("BatchPace", "Oracle BatchPace"))
        ]
        .groupby(["environment_id", "budget"], as_index=False)["mean"]
        .max()
        .rename(columns={"mean": "best_competitor_gain"})
    )
    per_budget = batchpace.merge(competitor, on=["environment_id", "budget"])
    per_budget["margin"] = (
        per_budget["batchpace_gain"] - per_budget["best_competitor_gain"]
    )
    ranking = (
        per_budget.groupby(
            ["environment_id", "environment_number", "alpha", "beta"],
            as_index=False,
        )
        .agg(
            mean_batchpace_gain=("batchpace_gain", "mean"),
            mean_margin=("margin", "mean"),
            winning_budgets=("margin", lambda values: int((values > 1e-9).sum())),
            minimum_margin=("margin", "min"),
            maximum_margin=("margin", "max"),
        )
        .sort_values(
            ["mean_margin", "winning_budgets", "environment_number"],
            ascending=[False, False, True],
        )
        .head(12)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def make_figure(summary: pd.DataFrame, ranking: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(30.0, 21.0), sharex=False, sharey=False)
    handles = []
    labels = []

    for panel_index, (ax, rank_row) in enumerate(zip(axes.flat, ranking.itertuples())):
        data = summary.loc[summary["environment_id"].eq(rank_row.environment_id)]
        for method in METHODS:
            subset = data.loc[data["method"].eq(method)].sort_values("budget")
            style = STYLES[method]
            x = subset["budget"].to_numpy()
            y = subset["mean"].to_numpy()
            ci = subset["ci95"].to_numpy()
            linewidth = style["linewidth"] + (0.8 if method == "BatchPace" else 0.4)
            markersize = style["markersize"] + (1.5 if method == "BatchPace" else 1.0)
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
            if panel_index == 0:
                handles.append(line)
                labels.append(method)

        lower = (data["mean"] - data["ci95"]).min()
        upper = (data["mean"] + data["ci95"]).max()
        span = max(float(upper - lower), 1.0)
        ax.set_ylim(min(0.0, float(lower)) - 0.04 * span, float(upper) + 0.07 * span)
        ax.set_xlim(-10, 202)
        ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 192])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.20)
        ax.tick_params(labelsize=14, labelbottom=True)
        panel = chr(ord("a") + panel_index)
        ax.set_title(
            f"({panel}) Rank {rank_row.rank}: "
            f"Beta(α={rank_row.alpha:g}, β={rank_row.beta:g})\n"
            f"Mean BatchPace margin Δ={rank_row.mean_margin:+.3f}",
            fontsize=16.5,
            pad=13,
        )

    fig.suptitle(
        "Top 12 Stationary Beta Environments for BatchPace",
        fontsize=24,
        y=0.992,
    )
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
        "Ranked by mean gain margin over the strongest non-Oracle baseline, "
        "averaged across B>0; panel-specific y-axis ranges",
        ha="center",
        fontsize=14,
    )
    fig.supxlabel("Global budget (B)", fontsize=21, y=0.020)
    fig.supylabel("Total realized gain", fontsize=21, x=0.006)
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
    summary = pd.read_csv(SUMMARY)
    ranking = rank_environments(summary)
    if len(ranking) != 12:
        raise ValueError(f"Expected 12 selected environments, found {len(ranking)}")
    selected_ids = set(ranking["environment_id"])
    figure_data = summary.loc[summary["environment_id"].isin(selected_ids)].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT / "selection_ranking.csv", index=False)
    (OUTPUT / "selection_ranking.json").write_text(
        json.dumps(ranking.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    figure_data.to_json(
        OUTPUT / "figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    figure = make_figure(summary, ranking)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT / f"stationary_beta_top12_3x4.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(figure)
    print(ranking.to_string(index=False))
    print(f"Created top-12 3x4 panel in {OUTPUT}")


if __name__ == "__main__":
    main()
