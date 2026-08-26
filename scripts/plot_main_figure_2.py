"""Plot Main Figure 2 from the replay summary CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "real_llm" / "policy_summary.csv"
PNG = ROOT / "output" / "figures" / "fig2_real_budget.png"
PDF = ROOT / "output" / "pdf" / "fig2_real_budget.pdf"

METHODS = [
    "Offline Oracle",
    "BatchPace",
    "WISERouter-Offline",
    "Robust Batch-Level",
    "Sequential",
]
STYLES = {
    "Offline Oracle": dict(color="#377eb8", marker="o", linestyle="--", linewidth=1.7),
    "BatchPace": dict(color="#e66101", marker="s", linestyle="-", linewidth=2.2),
    "WISERouter-Offline": dict(color="#5e3c99", marker="D", linestyle="-.", linewidth=1.7),
    "Robust Batch-Level": dict(color="#1b9e77", marker="^", linestyle="--", linewidth=1.7),
    "Sequential": dict(color="#4d4d4d", marker="x", linestyle=":", linewidth=1.8),
}


def main() -> None:
    data = pd.read_csv(INPUT)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), sharey=True)
    panels = [("random", "Random stream"), ("category-bursty", "Category-bursty stream")]
    handles = []
    labels = []
    for axis, (environment, title) in zip(axes, panels, strict=True):
        panel = data[data["environment"].eq(environment)]
        for method in METHODS:
            rows = panel[panel["method"].eq(method)].sort_values("budget_percent")
            style = STYLES[method]
            line = axis.errorbar(
                rows["budget_percent"],
                rows["mean_delta_accuracy_pp"],
                yerr=rows["ci95_delta_accuracy_pp"],
                markersize=4.5,
                capsize=2.5,
                elinewidth=0.9,
                label=method,
                **style,
            )
            if axis is axes[0]:
                handles.append(line)
                labels.append(method)
        axis.set_title(title, pad=5)
        axis.set_xlabel("Strong-model budget K/N (%)")
        axis.set_xticks([5, 10, 20])
        axis.set_xlim(4, 21)
        axis.set_ylim(0, 21)
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy gain over cheap-only (percentage points)")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    figure.suptitle(
        "Real-LLM budget-performance curve (MMLU-Pro, Qwen3-0.6B to Qwen3-4B)",
        y=1.09,
        fontsize=10.5,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.005,
        "Mean and 95% CI across 5 stratified split means; each split averages 20 order seeds. N=500, batch size=20.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.93), w_pad=1.3)
    figure.savefig(PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
