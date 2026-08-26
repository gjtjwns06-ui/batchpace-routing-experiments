"""Create the publication-ready Main Figure 1 from experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "synthetic_main_figure_1" / "summary.json"
PDF_OUT = ROOT / "output" / "pdf" / "fig1_synthetic_controlled_study.pdf"
PNG_OUT = ROOT / "output" / "figures" / "fig1_synthetic_controlled_study.png"

METHOD_STYLE = {
    "Oracle BatchPace": dict(color="#2563EB", marker="o", linestyle="--"),
    "Predicted BatchPace": dict(color="#EA580C", marker="s", linestyle="-"),
    "Sequential": dict(color="#15803D", marker="^", linestyle="-."),
    "Fixed Threshold": dict(color="#DC2626", marker="D", linestyle=":"),
    "Equal Split": dict(color="#7C3AED", marker="v", linestyle="-"),
}


def get_rows(summary, panel, method):
    return sorted(
        [
            row for row in summary
            if row["panel"] == panel
            and row["environment"] == "bursty"
            and row["method"] == method
        ],
        key=lambda row: row["epsilon"] if panel == "fig1a" else row["batch_size"],
    )


def plot_series(ax, rows, method, x_key):
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row["mean_oracle_recovery"] for row in rows], dtype=float)
    yerr = np.asarray([row["ci95_oracle_recovery"] for row in rows], dtype=float)
    style = METHOD_STYLE[method]
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        label=method,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=1.6,
        markersize=4.5,
        markeredgewidth=0.7,
        capsize=2.2,
        elinewidth=0.9,
        zorder=3,
    )
    return y - yerr


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharey=True)
    lower_values = []

    methods_a = [
        "Oracle BatchPace",
        "Predicted BatchPace",
        "Sequential",
        "Fixed Threshold",
        "Equal Split",
    ]
    for method in methods_a:
        lower_values.extend(plot_series(axes[0], get_rows(summary, "fig1a", method), method, "epsilon"))

    methods_b = ["Predicted BatchPace", "Sequential", "Equal Split"]
    for method in methods_b:
        lower_values.extend(plot_series(axes[1], get_rows(summary, "fig1b", method), method, "batch_size"))

    axes[0].set_title("(a) Prediction-error robustness")
    axes[0].set_xlabel(r"Prediction error $\epsilon$")
    axes[0].set_ylabel("Offline-oracle recovery")
    axes[0].set_xticks([0.0, 0.05, 0.10, 0.20, 0.30, 0.40])
    axes[0].set_xticklabels(["0", ".05", ".10", ".20", ".30", ".40"])

    axes[1].set_title("(b) Benefit of observable batch size")
    axes[1].set_xlabel(r"Batch size $m$")
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 5, 10, 20, 50, 100])
    axes[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[1].xaxis.set_minor_locator(mpl.ticker.NullLocator())

    lower = max(0.0, np.floor((min(lower_values) - 0.04) * 10.0) / 10.0)
    for ax in axes:
        ax.set_ylim(lower, 1.02)
        ax.grid(True, color="#CBD5E1", linewidth=0.55, alpha=0.65)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.15,
    )
    fig.text(
        0.5,
        0.012,
        "Bursty workload; points show means over 250 episodes and bars show 95% confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.20, top=0.78, wspace=0.17)

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"pdf": str(PDF_OUT), "png": str(PNG_OUT)}, indent=2))


if __name__ == "__main__":
    main()
