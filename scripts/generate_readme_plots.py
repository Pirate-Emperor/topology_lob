from __future__ import annotations

"""Build a gallery of measured FI-2010 research plots for the README.

All plots are derived from committed experiment result JSON files, except for
the two training-prefix stationarity points documented in
results/RESUME_TARGET_VALIDATION.md. The figures are plain Matplotlib charts,
not generated artwork. Run from the repository root:
    python scripts/generate_readme_plots.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A100_PATH = ROOT / "results" / "a100_fi2010_validation_summary.json"
CPU_PATH = ROOT / "results" / "fi2010_validation_summary.json"
OUT = ROOT / "assets" / "figures"

# These two points are reported in results/RESUME_TARGET_VALIDATION.md.
# The machine-readable A100 summary stores the selected order but not the full
# order sweep, so keep these measured values explicit and label them as such.
STATIONARITY_SWEEP = (
    {"d": 0.2, "median_adf_p": 0.015357, "price_correlation": 0.995689},
    {"d": 0.3, "median_adf_p": 0.002453, "price_correlation": 0.979516},
)
TARGET_IC = 0.064

# Use different, built-in Matplotlib styles while keeping the charts simple.
PLOTS = {
    "readme_daily_ic.svg": "seaborn-v0_8-whitegrid",
    "readme_daily_auc.svg": "ggplot",
    "readme_ffd_throughput.svg": "bmh",
    "readme_feature_ablation.svg": "tableau-colorblind10",
    "readme_robustness.svg": "seaborn-v0_8-paper",
    "readme_return_regression.svg": "seaborn-v0_8-muted",
    "readme_stationarity.svg": "classic",
}


def _save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        OUT / filename,
        format="svg",
        bbox_inches="tight",
        metadata={"Title": filename.replace("_", " ").removesuffix(".svg")},
    )
    plt.close(fig)


def plot_daily_ic(daily: list[dict]) -> None:
    with plt.style.context(PLOTS["readme_daily_ic.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        days = [f"Day {row['day']}" for row in daily]
        x = np.arange(len(days))
        pearson = [row["pearson_ic"] for row in daily]
        rank = [row["rank_ic"] for row in daily]
        ax.plot(x, pearson, marker="o", markersize=6, linewidth=2.2, label="Pearson IC")
        ax.plot(x, rank, marker="s", markersize=5, linewidth=2.0, label="Rank IC")
        ax.axhline(TARGET_IC, color="black", linestyle="--", linewidth=1.4, label="Target = 0.064")
        for xpos, value in zip(x, pearson):
            ax.annotate(f"{value:.4f}", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, days)
        ax.set_ylim(0, max(0.19, max(pearson + rank) + 0.025))
        ax.set(title="Fixed Focal Loss: daily held-out IC", xlabel="Held-out file", ylabel="Information coefficient (IC)")
        ax.legend(frameon=True, ncols=3, fontsize=8, loc="upper center")
        _save(fig, "readme_daily_ic.svg")


def plot_daily_auc(daily: list[dict]) -> None:
    with plt.style.context(PLOTS["readme_daily_auc.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        days = [f"Day {row['day']}" for row in daily]
        x = np.arange(len(days))
        values = [row["roc_auc"] for row in daily]
        bars = ax.bar(x, values, width=0.58, color="#1f77b4", edgecolor="#333333", linewidth=0.5)
        ax.axhline(0.5, color="#ff7f0e", linestyle="--", linewidth=1.5, label="Chance reference = 0.50")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xticks(x, days)
        ax.set_ylim(0.5, 0.78)
        ax.set(title="Fixed Focal Loss: daily ROC-AUC", xlabel="Held-out file", ylabel="ROC-AUC")
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, "readme_daily_auc.svg")


def plot_ffd_throughput(bench: dict) -> None:
    with plt.style.context(PLOTS["readme_ffd_throughput.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        cpu = bench["cpu_events_per_second"] / 1e6
        cuda = bench["cuda_events_per_second"] / 1e6
        bars = ax.barh(["CPU", "A100 CUDA"], [cpu, cuda], color=["#ff7f0e", "#1f77b4"], edgecolor="#333333", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 100)
        ax.set_xlabel("Million events per second (log scale)")
        ax.set_title("Fractional-differentiation throughput")
        ax.bar_label(bars, labels=[f"{cpu:.3f} M/s", f"{cuda:.2f} M/s"], padding=5, fontsize=9)
        ax.text(0.98, 0.1, f"{bench['speedup_vs_cpu']:.1f}× measured speedup", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold")
        ax.text(0.98, 0.02, f"200k events · width {bench['width']} · d={bench['d']}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        _save(fig, "readme_ffd_throughput.svg")


def plot_feature_ablation(ablation: dict) -> None:
    order = [
        ("micro_only", "Microstructure"),
        ("micro_plus_fd", "Micro + FFD"),
        ("micro_plus_topology", "Micro + topology"),
        ("full", "Full stack"),
    ]
    with plt.style.context(PLOTS["readme_feature_ablation.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        x = np.arange(len(order))
        width = 0.36
        pearson = [ablation[key]["pearson_ic"] for key, _ in order]
        rank = [ablation[key]["rank_ic"] for key, _ in order]
        ax.bar(x - width / 2, pearson, width, label="Pearson IC")
        ax.bar(x + width / 2, rank, width, label="Rank IC")
        ax.set_xticks(x, [name for _, name in order], rotation=10, ha="right")
        ax.set_ylim(0, max(0.20, max(pearson + rank) + 0.02))
        ax.set(title="Feature ablation — CPU reference run", xlabel="Feature group", ylabel="Information coefficient (IC)")
        ax.legend(fontsize=8, ncols=2)
        _save(fig, "readme_feature_ablation.svg")


def plot_robustness(focal: dict) -> None:
    pooled = focal["pooled_pearson_ic"]
    rob = focal["sequence_robustness"]
    lo, hi = rob["pearson_ic_ci_95"]
    with plt.style.context(PLOTS["readme_robustness.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.axvline(TARGET_IC, color="black", linestyle="--", linewidth=1.4, label="Target = 0.064")
        ax.errorbar(
            [pooled],
            [1],
            xerr=[[pooled - lo], [hi - pooled]],
            fmt="o",
            markersize=7,
            capsize=6,
            linewidth=2,
            color="#1f77b4",
            label="Pooled Pearson IC",
        )
        ax.set_yticks([1], ["A100 fixed Focal"])
        ax.set_xlim(0, 0.19)
        ax.set_ylim(0.65, 1.35)
        ax.set_xlabel("Pooled Pearson IC")
        ax.set_title("Moving-block bootstrap interval")
        ax.text(pooled, 1.12, f"{pooled:.4f}", ha="center", fontsize=9)
        ax.text((lo + hi) / 2, 0.82, f"95% CI [{lo:.4f}, {hi:.4f}]", ha="center", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.02, 0.03, f"{rob['moving_block_bootstrap_replicates']} resamples · block size {rob['moving_block_size_events']} events · placebo p={rob['circular_shift_two_sided_p']:.3f}", transform=ax.transAxes, fontsize=7.5)
        _save(fig, "readme_robustness.svg")


def plot_return_regression(daily: list[dict]) -> None:
    with plt.style.context(PLOTS["readme_return_regression.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        days = [f"Day {row['day']}" for row in daily]
        x = np.arange(len(days))
        width = 0.36
        pearson = [row["pearson_ic"] for row in daily]
        rank = [row["rank_ic"] for row in daily]
        ax.bar(x - width / 2, pearson, width, label="Pearson IC")
        ax.bar(x + width / 2, rank, width, label="Rank IC")
        ax.set_xticks(x, days)
        ax.set_ylim(0, max(0.29, max(pearson + rank) + 0.03))
        ax.set(title="Walk-forward selected return regressor", xlabel="Held-out file", ylabel="Information coefficient (IC)")
        ax.legend(fontsize=8, ncols=2)
        ax.text(0.02, 0.96, "Separate objective from the Focal Loss classifier", transform=ax.transAxes, va="top", fontsize=8)
        _save(fig, "readme_return_regression.svg")


def plot_stationarity() -> None:
    rows = list(STATIONARITY_SWEEP)
    labels = [f"d = {row['d']:.1f}" for row in rows]
    with plt.style.context(PLOTS["readme_stationarity.svg"]), matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
        x = np.arange(len(rows))
        pvalues = [row["median_adf_p"] for row in rows]
        correlations = [row["price_correlation"] for row in rows]
        bars = axes[0].bar(x, pvalues, color=["#ff7f0e", "#1f77b4"], edgecolor="#333333", linewidth=0.5)
        axes[0].set_yscale("log")
        axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axes[0].set_xticks(x, labels)
        axes[0].set(title="Median ADF p-value", ylabel="p-value (lower indicates stronger stationarity evidence)")
        axes[1].bar(x, correlations, color=["#ff7f0e", "#1f77b4"], edgecolor="#333333", linewidth=0.5)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.95, 1.01)
        axes[1].set(title="Correlation with original price", ylabel="Correlation (higher retains more)")
        fig.suptitle("Training-only stationarity / price-retention trade-off", fontsize=12)
        _save(fig, "readme_stationarity.svg")


def main() -> None:
    a100 = json.loads(A100_PATH.read_text(encoding="utf-8"))
    cpu = json.loads(CPU_PATH.read_text(encoding="utf-8"))
    focal = a100["fixed_focal"]
    plot_daily_ic(focal["daily"])
    plot_daily_auc(focal["daily"])
    plot_ffd_throughput(a100["ffd_microbenchmark"])
    plot_feature_ablation(cpu["ablation"])
    plot_robustness(focal)
    plot_return_regression(a100["continuous_return_regression"]["heldout_by_day"])
    plot_stationarity()
    missing = [name for name in PLOTS if not (OUT / name).is_file() or (OUT / name).stat().st_size < 1000]
    if missing:
        raise RuntimeError(f"Missing or unexpectedly small plot output(s): {missing}")
    print(f"Generated {len(PLOTS)} README plots in {OUT}")
    for name in PLOTS:
        print(f"  {name}: {(OUT / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()
