from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "fi2010_validation_summary.json"
OUT = ROOT / "assets" / "figures" / "fi2010_validation_dashboard.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    days = data["days"]
    ablation = data["ablation"]
    names = list(ablation)
    sweep = data["ic_sweep"]
    fixed = sweep["heldout_pooled"]["fixed_reference_focal_threshold_0_0005"]
    selected = sweep["heldout_pooled"]["validation_selected_logloss_threshold_0"]
    selection = sweep["selected"]

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    x = np.array([d["day"] for d in days])

    ax[0, 0].plot(x, [d["pearson_ic"] for d in days], marker="o", label="Pearson IC")
    ax[0, 0].plot(x, [d["rank_ic"] for d in days], marker="o", label="Rank IC")
    ax[0, 0].axhline(0.064, linestyle="--", color="black", label="0.064 target")
    ax[0, 0].set_title("Fixed Focal: daily held-out IC")
    ax[0, 0].set_xlabel("Held-out file")
    ax[0, 0].set_ylabel("IC")
    ax[0, 0].legend()

    ax[0, 1].plot(x, [d["roc_auc"] for d in days], marker="o")
    ax[0, 1].set_title("Fixed Focal: daily ROC-AUC")
    ax[0, 1].set_xlabel("Held-out file")
    ax[0, 1].set_ylabel("ROC-AUC")
    ax[0, 1].set_ylim(0.5, 0.8)

    groups = ["Fixed Focal", "Train-only selected"]
    xx = np.arange(len(groups))
    ax[0, 2].bar(xx - 0.18, [fixed["pearson_ic"], selected["pearson_ic"]], 0.36, label="Pearson IC")
    ax[0, 2].bar(xx + 0.18, [fixed["rank_ic"], selected["rank_ic"]], 0.36, label="Rank IC")
    ax[0, 2].axhline(0.064, linestyle="--", color="black", label="0.064 target")
    ax[0, 2].set_xticks(xx, groups, rotation=10, ha="right")
    ax[0, 2].set_title("Held-out IC: fixed vs control")
    ax[0, 2].set_ylabel("IC")
    ax[0, 2].legend()

    xx = np.arange(len(names))
    ax[1, 0].bar(xx - 0.18, [ablation[n]["pearson_ic"] for n in names], 0.36, label="Pearson IC")
    ax[1, 0].bar(xx + 0.18, [ablation[n]["rank_ic"] for n in names], 0.36, label="Rank IC")
    ax[1, 0].axhline(0.0, linewidth=1)
    ax[1, 0].set_xticks(xx, names, rotation=25, ha="right")
    ax[1, 0].set_title("Feature ablation: IC")
    ax[1, 0].set_ylabel("IC")
    ax[1, 0].legend()

    ax[1, 1].bar(xx, [ablation[n]["roc_auc"] for n in names])
    ax[1, 1].set_xticks(xx, names, rotation=25, ha="right")
    ax[1, 1].set_title("Feature ablation: ROC-AUC")
    ax[1, 1].set_ylabel("ROC-AUC")
    ax[1, 1].set_ylim(0.68, 0.72)

    val = selection["validation_pearson_ic"]
    val_rank = selection["validation_rank_ic"]
    categories = ["Validation", "Held-out"]
    ax[1, 2].bar([0, 1], [val, selected["pearson_ic"]], width=0.36, label="Pearson IC")
    ax[1, 2].bar([0.36, 1.36], [val_rank, selected["rank_ic"]], width=0.36, label="Rank IC")
    ax[1, 2].set_xticks([0.18, 1.18], categories)
    ax[1, 2].set_title("Train-only selection vs final test")
    ax[1, 2].set_ylabel("IC")
    ax[1, 2].legend()

    split = data["sequence_splitting"]
    fig.suptitle(
        "FI-2010 — corrected sequence-safe results "
        f"({split['inferred_sequences_per_file'][0]} inferred sequences/file; "
        f"{split['total_cuts']} cuts; CPU FFD in this run)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
