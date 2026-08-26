"""Select and plot the eight stationary LLM environments best for BatchPace."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from plot_stationary_llm_suite import METHODS, STYLES, summarize


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "stationary_llm_suite"
OUTPUT = ROOT / "output" / "stationary_llm_top8"


def rank_environments(metrics: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Rank by aggregate mean advantage over the strongest plotted baseline.

    For each environment and method, gain is first averaged over all seeds and
    all nonzero budgets.  The margin is BatchPace's aggregate mean minus the
    largest aggregate mean among the four competing policies.  Average margin
    over all four competitors and BatchPace gain are deterministic tie-breakers.
    """
    selected = metrics.loc[
        metrics["budget"].gt(0) & metrics["method"].isin(METHODS)
    ]
    aggregate = selected.groupby(
        ["environment_id", "method"], sort=True
    )["total_realized_gain"].mean().unstack("method")
    competitors = [method for method in METHODS if method != "BatchPace"]

    ranking = pd.DataFrame({
        "mean_batchpace_gain": aggregate["BatchPace"],
        "best_competitor_gain": aggregate[competitors].max(axis=1),
        "mean_competitor_gain": aggregate[competitors].mean(axis=1),
    }).reset_index()
    ranking["mean_margin"] = (
        ranking["mean_batchpace_gain"] - ranking["best_competitor_gain"]
    )
    ranking["mean_margin_over_average_competitor"] = (
        ranking["mean_batchpace_gain"] - ranking["mean_competitor_gain"]
    )
    ranking = ranking.merge(
        specs[["environment_id", "category", "source", "subject_label"]],
        on="environment_id",
        validate="one_to_one",
    )
    ranking = ranking.sort_values(
        ["mean_margin", "mean_margin_over_average_competitor", "mean_batchpace_gain", "environment_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).head(8).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def make_figure(summary: pd.DataFrame, ranking: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(31.0, 15.8), sharex=False, sharey=False)
    handles: list[object] = []
    labels: list[str] = []

    for panel_index, (axis, rank_row) in enumerate(zip(axes.flat, ranking.itertuples())):
        data = summary.loc[summary["environment_id"].eq(rank_row.environment_id)]
        for method in METHODS:
            subset = data.loc[data["method"].eq(method)].sort_values("budget")
            style = STYLES[method]
            x = subset["budget"].to_numpy()
            y = subset["mean"].to_numpy()
            ci = subset["ci95"].to_numpy()
            (line,) = axis.plot(
                x,
                y,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"] + (0.8 if method == "BatchPace" else 0.3),
                markersize=style["markersize"] + (1.3 if method == "BatchPace" else 0.7),
                alpha=style["alpha"],
                zorder=style["zorder"],
            )
            axis.fill_between(
                x,
                y - ci,
                y + ci,
                color=style["color"],
                alpha=0.17 if method == "BatchPace" else 0.06,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
            if panel_index == 0:
                handles.append(line)
                labels.append(method)

        lower = float((data["mean"] - data["ci95"]).min())
        upper = float((data["mean"] + data["ci95"]).max())
        span = max(1.0, upper - lower)
        axis.set_ylim(min(0.0, lower) - 0.04 * span, max(0.0, upper) + 0.07 * span)
        axis.set_xlim(-10, 202)
        axis.set_xticks([0, 24, 48, 72, 96, 120, 144, 192])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        axis.grid(True, alpha=0.20)
        axis.tick_params(labelsize=14, labelbottom=True)
        panel = chr(ord("a") + panel_index)
        axis.set_title(
            f"({panel}) Rank {rank_row.rank}: {rank_row.subject_label}\n"
            f"{str(rank_row.category).title()} | Mean BatchPace margin Δ={rank_row.mean_margin:+.3f}",
            fontsize=16.0,
            pad=13,
        )

    fig.suptitle(
        "Top 8 Stationary LLM-derived Environments for BatchPace",
        fontsize=25,
        y=0.992,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=5,
        fontsize=16,
        frameon=True,
    )
    fig.text(
        0.5,
        0.904,
        "Ranked by mean gain margin over the strongest aggregate competitor, "
        "averaged across B>0; panel-specific y-axis ranges",
        ha="center",
        fontsize=14,
    )
    fig.supxlabel("Global budget (B)", fontsize=22, y=0.018)
    fig.supylabel("Total realized gain", fontsize=22, x=0.007)
    fig.subplots_adjust(
        left=0.055,
        right=0.994,
        bottom=0.080,
        top=0.845,
        wspace=0.18,
        hspace=0.34,
    )
    return fig


def main() -> None:
    metrics = pd.read_parquet(RESULT / "metrics.parquet")
    specs = pd.DataFrame(json.loads(
        (RESULT / "environment_manifest.json").read_text(encoding="utf-8")
    ))
    ranking = rank_environments(metrics, specs)
    if len(ranking) != 8:
        raise ValueError(f"expected 8 selected environments, found {len(ranking)}")
    summary = summarize(metrics)
    selected_ids = set(ranking["environment_id"])
    figure_data = summary.loc[summary["environment_id"].isin(selected_ids)].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT / "selection_ranking.csv", index=False)
    ranking.to_json(
        OUTPUT / "selection_ranking.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )
    figure_data.to_json(
        OUTPUT / "figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    figure = make_figure(summary, ranking)
    stem = OUTPUT / "stationary_llm_top8_2x4"
    figure.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    print(ranking.to_string(index=False), flush=True)
    print(f"created {stem}.png/.svg/.pdf", flush=True)


if __name__ == "__main__":
    main()

