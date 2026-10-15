from __future__ import annotations

"""Training-only, walk-forward-selected FI-2010 continuous-return experiment.

The model configuration is selected from five chronological validation folds
within the training file. The held-out files are scored only after selection.
No raw data are written to the repository.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import sys
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fi2010_return_regression import (
    FEATURE_SETS,
    build_features,
    fit_regressor,
    forward_returns,
    score,
)

DATASET_PAGE = "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
# fit_end, validation_start, validation_end as fractions of the training file.
FOLDS = ((0.45, 0.50, 0.60), (0.55, 0.60, 0.70), (0.65, 0.70, 0.80),
         (0.75, 0.80, 0.90), (0.85, 0.90, 0.99))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_indices(
    indices: np.ndarray,
    valid: np.ndarray,
    *,
    train_events: int,
    horizon: int,
    purge: int,
    fractions: tuple[float, float, float],
    sequence_boundaries: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a purged forward fold within every stock's own time sequence."""
    fit_fraction, val_start_fraction, val_end_fraction = fractions
    if not 0 < fit_fraction < val_start_fraction < val_end_fraction < 1:
        raise ValueError("fold fractions must satisfy 0 < fit < val_start < val_end < 1")
    idx = np.asarray(indices, dtype=int)
    is_valid = np.asarray(valid, dtype=bool)
    if idx.ndim != 1 or is_valid.ndim != 1 or len(idx) != len(is_valid):
        raise ValueError("indices and valid mask must be matching 1-D arrays")

    gap = max(purge, horizon)
    fit = np.zeros(len(idx), dtype=bool)
    val = np.zeros(len(idx), dtype=bool)
    used = 0
    for start, end in sequence_boundaries:
        if end <= start or start < 0 or end > train_events:
            if start >= train_events:
                continue
            raise ValueError(f"invalid training stock sequence: {(start, end)}")
        length = end - start
        fit_end = start + int(length * fit_fraction)
        val_start = start + int(length * val_start_fraction)
        val_end = start + int(length * val_end_fraction)
        fit |= is_valid & (idx >= start) & (idx < fit_end - gap)
        val |= is_valid & (idx >= val_start + gap) & (idx < val_end - horizon)
        used += 1
    if used == 0:
        raise ValueError("no training stock sequences found for validation")
    return np.flatnonzero(fit), np.flatnonzero(val)

def aggregate_fold_scores(folds: list[dict[str, Any]]) -> dict[str, float]:
    if not folds:
        raise ValueError("at least one fold score is required")
    pearson = np.asarray([x["pearson_ic"] for x in folds], dtype=float)
    rank = np.asarray([x["rank_ic"] for x in folds], dtype=float)
    if not np.isfinite(pearson).all() or not np.isfinite(rank).all():
        raise ValueError("fold IC values must be finite")
    return {
        "fold_count": int(len(folds)),
        "mean_pearson_ic": float(np.mean(pearson)),
        "std_pearson_ic": float(np.std(pearson, ddof=1)) if len(pearson) > 1 else 0.0,
        "mean_rank_ic": float(np.mean(rank)),
        "std_rank_ic": float(np.std(rank, ddof=1)) if len(rank) > 1 else 0.0,
        "positive_pearson_folds": int(np.sum(pearson > 0)),
        "validation_rows_total": int(sum(x["rows"] for x in folds)),
    }


def dashboard(path: Path, candidates: list[dict], selected: dict, day_rows: list[dict], target: float) -> None:
    by_feature = {}
    for name in FEATURE_SETS:
        options = [x for x in candidates if x["feature_set"] == name]
        if options:
            by_feature[name] = max(options, key=lambda x: x["validation"]["mean_pearson_ic"])
    names = list(by_feature)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    means = [by_feature[n]["validation"]["mean_pearson_ic"] for n in names]
    axes[0].bar(names, means)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axhline(target, color="darkred", linestyle="--", label=f"Target {target:.3f}")
    axes[0].set_title("Walk-forward validation: best IC by feature set")
    axes[0].set_ylabel("Mean Pearson IC over validation folds")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    days = [x["day"] for x in day_rows]
    pearson = [x["metrics"]["pearson_ic"] for x in day_rows]
    rank = [x["metrics"]["rank_ic"] for x in day_rows]
    x = np.arange(len(days))
    axes[1].bar(x - 0.18, pearson, width=0.36, label="Pearson IC")
    axes[1].bar(x + 0.18, rank, width=0.36, label="Rank IC")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(target, color="darkred", linestyle="--", label=f"Target {target:.3f}")
    axes[1].set_xticks(x, [f"Test {d}" for d in days])
    axes[1].set_title(f"Untouched holdout | {selected['horizon']} events, {selected['feature_set']}")
    axes[1].set_ylabel("Information coefficient")
    axes[1].legend()
    fig.suptitle("FI-2010: walk-forward-selected return regression", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = build_features(args)
    train_events = int(bundle["train_events"])
    micro = bundle["micro"]
    boundaries = bundle["boundaries"]
    raw_indices = np.arange(len(micro))
    feature_arrays = bundle["feature_arrays"]
    grid = [(h, name) for h in args.horizons for name in FEATURE_SETS]
    candidate_rows: list[dict[str, Any]] = []

    # Model selection uses only forward-chaining validation folds inside train.
    for horizon, feature_set in grid:
        X_all, feature_names = feature_arrays[feature_set]
        future = forward_returns(micro["log_mid"].to_numpy(), boundaries, horizon)
        valid = np.isfinite(X_all).all(axis=1) & np.isfinite(future) & (raw_indices < train_events)
        fold_rows = []
        for fold_id, fractions in enumerate(FOLDS, start=1):
            fit_pos, val_pos = fold_indices(
                raw_indices, valid,
                train_events=train_events, horizon=horizon, purge=args.purge,
                fractions=fractions,
                sequence_boundaries=boundaries,
            )
            if len(fit_pos) < args.minimum_fit_rows or len(val_pos) < args.minimum_validation_rows:
                raise RuntimeError(
                    f"insufficient rows in fold {fold_id} h={horizon} {feature_set}: "
                    f"fit={len(fit_pos)}, validation={len(val_pos)}"
                )
            model = fit_regressor(
                X_all[fit_pos], future[fit_pos],
                seed=args.seed + horizon, rounds=args.rounds, nthread=args.nthread,
            )
            prediction = model.predict(__import__("xgboost").DMatrix(X_all[val_pos])) / 1e4
            metrics = score(prediction, future[val_pos])
            fold_rows.append({
                "fold": fold_id,
                "fit_fraction": fractions[0],
                "validation_fraction_range": [fractions[1], fractions[2]],
                "fit_rows": int(len(fit_pos)),
                "rows": int(len(val_pos)),
                "metrics": metrics,
                "pearson_ic": metrics["pearson_ic"],
                "rank_ic": metrics["rank_ic"],
            })
        candidate_rows.append({
            "horizon": int(horizon),
            "feature_set": feature_set,
            "feature_count": int(X_all.shape[1]),
            "validation_folds": fold_rows,
            "validation": aggregate_fold_scores(fold_rows),
        })

    selected = max(
        candidate_rows,
        key=lambda r: (r["validation"]["mean_pearson_ic"], r["validation"]["mean_rank_ic"]),
    )
    horizon = int(selected["horizon"])
    feature_set = selected["feature_set"]
    X_all, feature_names = feature_arrays[feature_set]
    future = forward_returns(micro["log_mid"].to_numpy(), boundaries, horizon)
    purge = max(args.purge, horizon)
    final_train = np.isfinite(X_all).all(axis=1) & np.isfinite(future) & (raw_indices < train_events)
    for seq_start, seq_end in boundaries:
        if seq_end <= train_events:
            final_train[max(seq_start, seq_end - purge):seq_end] = False
    test_mask = np.isfinite(X_all).all(axis=1) & np.isfinite(future) & (raw_indices >= train_events)
    train_pos = np.flatnonzero(final_train)
    test_pos = np.flatnonzero(test_mask)
    if len(train_pos) < args.minimum_fit_rows or len(test_pos) < args.minimum_test_rows:
        raise RuntimeError(f"insufficient final rows train={len(train_pos)} test={len(test_pos)}")
    final_model = fit_regressor(
        X_all[train_pos], future[train_pos],
        seed=args.seed + horizon, rounds=args.rounds, nthread=args.nthread,
    )

    day_rows = []
    pooled_predictions, pooled_targets = [], []
    file_boundaries = bundle["file_boundaries"]
    for day, (start, end) in enumerate(file_boundaries[1:], start=8):
        pos = np.flatnonzero(test_mask & (raw_indices >= start) & (raw_indices < end))
        if len(pos) < args.minimum_test_rows // 3:
            raise RuntimeError(f"insufficient test rows for day {day}: {len(pos)}")
        predictions = final_model.predict(__import__("xgboost").DMatrix(X_all[pos])) / 1e4
        metrics = score(predictions, future[pos])
        day_rows.append({
            "day": day,
            "events": int(end - start),
            "rows": int(len(pos)),
            "metrics": metrics,
        })
        pooled_predictions.append(predictions)
        pooled_targets.append(future[pos])
    pooled_predictions = np.concatenate(pooled_predictions)
    pooled_targets = np.concatenate(pooled_targets)
    pooled = score(pooled_predictions, pooled_targets)

    input_files = [Path(args.train), *(Path(p) for p in args.test)]
    result = {
        "experiment": "FI-2010 continuous return regression with walk-forward validation selection",
        "dataset": {
            "name": "FI-2010",
            "record": DATASET_PAGE,
            "license": DATASET_LICENSE,
            "attribution": "Ntakaris et al., Journal of Forecasting 37(8), 852-866 (2018), DOI 10.1002/for.2543",
            "input_sha256": {p.name: sha256(p) for p in input_files},
            "train_events": train_events,
            "test_events": [int(len(f)) for f in bundle["frames"][1:]],
            "sequence_jump_threshold_pct": 5.0,
            "sequence_cuts": bundle["sequence_cut_records"],
            "stock_sequences_per_file": [
                int(sum(1 for b in boundaries if file_start <= b[0] and b[1] <= file_end))
                for file_start, file_end in file_boundaries
            ],
        },
        "method": {
            "target": f"log(mid[t+{horizon}])-log(mid[t])",
            "split": "five expanding chronological validation folds inside training file, purge >= horizon",
            "selection_metric": "mean Pearson IC across training-only validation folds; Rank IC used only as tie-break",
            "test_use": "selected configuration refit on training file; held-out files scored only after selection",
            "validation_folds": [{"fit_fraction": f, "validation_start_fraction": s, "validation_end_fraction": e} for f, s, e in FOLDS],
            "candidate_count": len(candidate_rows),
            "selected_d": bundle["selected_d"],
            "stationarity": bundle["stationarity"],
            "tda": bundle["tda"],
            "topology_clouds": bundle["topology_clouds"],
            "ffd_backend": bundle["ffd_backend"],
            "gpu_used": bundle["gpu_used"],
            "python": platform.python_version(),
            "rounds": args.rounds,
            "purge": args.purge,
        },
        "selection": {
            "horizon": horizon,
            "feature_set": feature_set,
            "feature_count": int(X_all.shape[1]),
            "feature_names": feature_names,
            "validation": selected["validation"],
            "candidate_rows": candidate_rows,
        },
        "heldout_days": day_rows,
        "heldout_pooled": pooled,
        "target_pearson_ic": args.target_ic,
        "target_status": {
            "pearson_ic_at_least_target": bool(pooled["pearson_ic"] >= args.target_ic),
            "rank_ic_at_least_target": bool(pooled["rank_ic"] >= args.target_ic),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.dashboard) if args.dashboard else out.with_suffix(".png")
    dashboard(image_path, candidate_rows, {"horizon": horizon, "feature_set": feature_set}, day_rows, args.target_ic)
    result["dashboard_path"] = str(image_path)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_return_regression_selection.json")
    parser.add_argument("--dashboard", default="results/fi2010_return_regression_selection.png")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 30, 50, 100, 200, 500, 1000])
    parser.add_argument("--purge", type=int, default=50)
    parser.add_argument("--ffd-width", type=int, default=256)
    parser.add_argument("--tda-stride", type=int, default=1000)
    parser.add_argument("--tda-jobs", type=int, default=-1)
    parser.add_argument("--stationarity-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2010)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--target-ic", type=float, default=0.064)
    parser.add_argument("--minimum-fit-rows", type=int, default=1000)
    parser.add_argument("--minimum-validation-rows", type=int, default=1000)
    parser.add_argument("--minimum-test-rows", type=int, default=1000)
    parser.add_argument("--require-gtda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not args.horizons or any(h < 1 for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        parser.error("horizons must be unique positive integers")
    if args.purge < 1 or args.rounds < 1 or args.nthread < 1 or args.ffd_width < 1:
        parser.error("purge, rounds, nthread, and ffd-width must be positive")
    if not np.isfinite(args.target_ic):
        parser.error("target-ic must be finite")
    run(args)


if __name__ == "__main__":
    main()
