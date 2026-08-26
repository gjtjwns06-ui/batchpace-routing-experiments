"""Plot the eight selected LLM environments with BatchPace emphasized."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "benchmark_suite" / "llm_metrics.parquet"
OUTPUT = ROOT / "output" / "selected_llm_highlighted"

# Paper figures are embedded at the full two-column width of the NeurIPS
# template.  STIXGeneral is metrically close to the Times-style body font and
# is bundled with Matplotlib, so the PDF remains portable on Overleaf.
mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

# Preserve the order requested for the deliverable.
SELECTED_NUMBERS = (14, 13, 12, 8, 7, 4, 2, 1)
ASCENDING_NUMBERS = tuple(sorted(SELECTED_NUMBERS))

# Oracle BatchPace and Sequential Pacing are intentionally excluded.
METHODS = (
    "BatchPace",
    "Fixed Threshold",
    "Equal Split",
    "WISERouter-Offline",
    "Robust Batch-Level Query Routing",
)

STYLES = {
    "BatchPace": {
        "color": "#D62728",
        "marker": "o",
        "linewidth": 3.2,
        "markersize": 7.0,
        "alpha": 1.0,
        "zorder": 6,
    },
    "Fixed Threshold": {
        "color": "#009E73",
        "marker": "^",
        "linewidth": 1.8,
        "markersize": 5.0,
        "alpha": 0.82,
        "zorder": 3,
    },
    "Equal Split": {
        "color": "#9467BD",
        "marker": "D",
        "linewidth": 1.8,
        "markersize": 4.8,
        "alpha": 0.82,
        "zorder": 3,
    },
    "WISERouter-Offline": {
        "color": "#1F77B4",
        "marker": "P",
        "linewidth": 1.8,
        "markersize": 5.0,
        "alpha": 0.82,
        "zorder": 3,
    },
    "Robust Batch-Level Query Routing": {
        "color": "#E69F00",
        "marker": "X",
        "linewidth": 1.8,
        "markersize": 5.0,
        "alpha": 0.82,
        "zorder": 3,
    },
}


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["method"].isin(METHODS)].copy()
    grouped = selected.groupby(
        ["environment_id", "budget", "method"], sort=True
    )["total_realized_gain"]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["ci95"] = 1.96 * result["std"].fillna(0) / np.sqrt(result["count"])
    return result


def selected_environment_ids(metrics: pd.DataFrame) -> list[str]:
    available = sorted(metrics["environment_id"].unique())
    result = []
    for number in SELECTED_NUMBERS:
        prefix = f"llm_{number:02d}_"
        matches = [name for name in available if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one environment for {prefix}, found {matches}")
        result.append(matches[0])
    return result


def make_figure(environment_id: str, data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)

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
            alpha=0.16 if method == "BatchPace" else 0.07,
            linewidth=0,
            zorder=style["zorder"] - 1,
        )

    ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
    ax.set_xlabel("Global budget (B)", fontsize=15)
    ax.set_ylabel("Total realized gain", fontsize=15)
    ax.set_title(f"LLM-derived environment: {environment_id}", fontsize=16)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=12, frameon=True, ncol=2)
    ax.tick_params(labelsize=12)
    return fig


def short_title(environment_id: str, panel_index: int) -> str:
    parts = environment_id.split("_")
    number = parts[1]
    source = parts[2].title()
    target = parts[4].title()
    regime = "IID" if environment_id.endswith("_iid") else "Batch-bursty"
    panel = chr(ord("a") + panel_index)
    return f"({panel}) {number} {source} → {target}\n{regime}"


def make_grid_figure(summary: pd.DataFrame, environment_ids: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 4.85),
        sharex=True,
        sharey=True,
    )

    handles = []
    labels = []
    for panel_index, (ax, environment_id) in enumerate(zip(axes.flat, environment_ids)):
        data = summary.loc[summary["environment_id"].eq(environment_id)]
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
                linewidth=2.05 if method == "BatchPace" else 1.15,
                markersize=4.1 if method == "BatchPace" else 3.0,
                alpha=style["alpha"],
                zorder=style["zorder"],
            )
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                color=style["color"],
                alpha=0.16 if method == "BatchPace" else 0.07,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
            if panel_index == 0:
                handles.append(line)
                labels.append(method)

        ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        ax.set_title(
            short_title(environment_id, panel_index),
            fontsize=8.7,
            linespacing=0.92,
            pad=4,
        )
        ax.grid(True, alpha=0.20)
        # ``sharex=True`` hides labels on the upper row by default.  Keep the
        # same numeric budget labels visible on all eight panels.
        ax.tick_params(axis="both", labelsize=8.1, length=2.8, pad=1.8)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xticks([0, 48, 96, 144, 192])

    lower = (summary["mean"] - summary["ci95"]).min()
    upper = (summary["mean"] + summary["ci95"]).max()
    span = upper - lower
    axes[0, 0].set_ylim(min(0.0, lower) - 0.03 * span, upper + 0.04 * span)

    fig.suptitle(
        "Selected LLM-derived environments: Budget–Gain Comparison",
        fontsize=12.2,
        y=0.988,
    )
    fig.supxlabel("Global budget (B)", fontsize=9.4, y=0.018)
    fig.supylabel("Total realized gain", fontsize=9.4, x=0.006)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.950),
        ncol=5,
        fontsize=7.8,
        frameon=True,
        handlelength=1.7,
        columnspacing=0.8,
        borderpad=0.30,
    )
    fig.subplots_adjust(left=0.077, right=0.995, bottom=0.105, top=0.805,
                        wspace=0.15, hspace=0.39)
    return fig


def main() -> None:
    metrics = pd.read_parquet(METRICS)
    environment_ids = selected_environment_ids(metrics)
    summary = summarize(metrics)
    selected_summary = summary.loc[summary["environment_id"].isin(environment_ids)]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_summary.to_json(
        OUTPUT / "selected_figure_data.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=15,
    )

    manifest_rows = []
    pdf_path = OUTPUT / "selected_llm_highlighted.pdf"
    with PdfPages(pdf_path) as pdf:
        for environment_id in environment_ids:
            data = selected_summary.loc[
                selected_summary["environment_id"].eq(environment_id)
            ]
            fig = make_figure(environment_id, data)
            png_path = OUTPUT / f"{environment_id}.png"
            svg_path = OUTPUT / f"{environment_id}.svg"
            fig.savefig(png_path, dpi=220)
            fig.savefig(svg_path)
            pdf.savefig(fig)
            plt.close(fig)
            manifest_rows.append(
                {
                    "environment_number": int(environment_id[4:6]),
                    "environment_id": environment_id,
                    "png": str(png_path.relative_to(ROOT)),
                    "svg": str(svg_path.relative_to(ROOT)),
                }
            )

    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "figure_manifest.csv", index=False)

    ascending_ids = []
    for number in ASCENDING_NUMBERS:
        prefix = f"llm_{number:02d}_"
        ascending_ids.append(next(name for name in environment_ids if name.startswith(prefix)))
    grid = make_grid_figure(selected_summary, ascending_ids)
    grid_png = OUTPUT / "selected_llm_2x4_ascending.png"
    grid_svg = OUTPUT / "selected_llm_2x4_ascending.svg"
    grid_pdf = OUTPUT / "selected_llm_2x4_ascending.pdf"
    grid.savefig(grid_png, dpi=220)
    grid.savefig(grid_svg)
    grid.savefig(grid_pdf)
    plt.close(grid)

    print(f"Created {len(manifest_rows)} selected figures in {OUTPUT}")
    print(f"Combined PDF: {pdf_path}")
    print(f"2x4 ascending panel: {grid_pdf}")


if __name__ == "__main__":
    main()
