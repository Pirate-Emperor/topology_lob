from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fi2010 import load_raw, raw40_to_canonical
from scripts.fi2010_feature_helpers import (
    fractional_diff_by_segment,
    microstructure_by_segment,
    topology_by_segment,
)
from topology_lob import select_stationary_d

DATASET_PAGE = "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
DATASET_LICENSE = "CC BY 4.0"
GRID = [(features, depth, rounds)
        for features in ("micro_only", "micro_plus_fd", "micro_plus_topology", "full")
        for depth in (2, 4)
        for rounds in (100, 220)]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson_ic(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(pred) != len(target) or len(pred) < 2:
        raise ValueError("pred and target must have the same length >= 2")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("pred and target must be finite")
    if np.std(pred) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def score_regression(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(pred) != len(target) or len(pred) == 0:
        raise ValueError("pred and target must have the same non-empty length")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("pred and target must be finite")
    rank = float(spearmanr(pred, target).statistic) if np.std(pred) and np.std(target) else 0.0
    return {
        "pearson_ic": pearson_ic(pred, target),
        "rank_ic": rank,
        "mae": float(np.mean(np.abs(pred - target))),
        "rmse": float(np.sqrt(np.mean(np.square(pred - target)))),
        "directional_accuracy": float(np.mean((pred > 0) == (target > 0))),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
    }


def build_regressor(X: np.ndarray, y: np.ndarray, *, depth: int, rounds: int, seed: int, nthread: int):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": depth,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 8,
        "lambda": 2.0,
        "seed": seed,
        "nthread": nthread,
    }
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds, verbose_eval=False)


def forward_returns(log_mid: np.ndarray, boundaries: list[tuple[int, int]], horizon: int) -> np.ndarray:
    values = np.asarray(log_mid, dtype=float)
    result = np.full(len(values), np.nan)
    for start, end in boundaries:
        if end - start <= horizon:
            continue
        result[start:end-horizon] = values[start+horizon:end] - values[start:end-horizon]
    return result


def make_dashboard(output: Path, rows: list[dict], selected: dict, days: list[dict], target: float = 0.064) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    best = {}
    for row in rows:
        name = row["features"]
        if name not in best or row["validation"]["pearson_ic"] > best[name]["validation"]["pearson_ic"]:
            best[name] = row
    names = list(best)
    values = [best[n]["validation"]["pearson_ic"] for n in names]
    axes[0].bar(names, values)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axhline(target, color="darkred", linestyle="--", label=f"IC target {target:.3f}")
    axes[0].set_title("Training-only validation: best IC by feature set")
    axes[0].set_ylabel("Pearson IC")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].legend()
    dx = np.arange(len(days))
    axes[1].bar(dx - 0.18, [d["metrics"]["pearson_ic"] for d in days], 0.36, label="Pearson IC")
    axes[1].bar(dx + 0.18, [d["metrics"]["rank_ic"] for d in days], 0.36, label="Rank IC")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(target, color="darkred", linestyle="--", label=f"IC target {target:.3f}")
    axes[1].set_xticks(dx, [f"Test {d['day']}" for d in days])
    axes[1].set_title(f"Final holdout: selected {selected['features']} regressor")
    axes[1].set_ylabel("Information coefficient")
    axes[1].legend()
    fig.suptitle("FI-2010 | return-regression experiment (validation selected)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    paths = [Path(args.train), *(Path(p) for p in args.test)]
    frames = [raw40_to_canonical(load_raw(path).features) for path in paths]
    train_events = len(frames[0])
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for frame in frames:
        boundaries.append((cursor, cursor + len(frame)))
        cursor += len(frame)
    total_events = cursor

    # Build causal microstructure features separately per file; do not roll over a day boundary.
    micro, _ = microstructure_by_segment(frames, depth_levels=5)
    train_log_mid = micro["log_mid"].to_numpy()[:train_events]
    d, stationarity = select_stationary_d(
        train_log_mid,
        np.linspace(0.0, 1.0, 11),
        0.05,
        args.ffd_width,
    )
    ffd, gpu_used, ffd_backend = fractional_diff_by_segment(
        micro["log_mid"].to_numpy(), boundaries, d, args.ffd_width,
        require_cuda=args.require_cuda,
    )
    micro["fracdiff"] = ffd

    radii = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    topology, topology_meta, topology_indices = topology_by_segment(
        frames, boundaries, radii,
        horizon=args.horizon, stride=args.tda_stride, levels=10,
        require_gtda=args.require_gtda, n_jobs=args.tda_jobs,
    )
    base = [c for c in micro.columns if c not in {"mid", "log_mid"}]
    micro_base = [c for c in base if c != "fracdiff"]
    matrices = {
        "micro_only": micro[micro_base].to_numpy(float),
        "micro_plus_fd": micro[base].to_numpy(float),
        "micro_plus_topology": np.hstack([micro[micro_base].to_numpy(float), topology.to_numpy(float)]),
        "full": np.hstack([micro[base].to_numpy(float), topology.to_numpy(float)]),
    }

    future = forward_returns(micro["log_mid"].to_numpy(), boundaries, args.horizon)
    valid = np.isfinite(future)
    for matrix in matrices.values():
        valid &= np.isfinite(matrix).all(axis=1)
    raw_index = np.flatnonzero(valid)
    train_pos = np.flatnonzero(raw_index < train_events)
    if len(train_pos) < 1000:
        raise RuntimeError(f"Too few usable training rows: {len(train_pos)}")
    split = int(len(train_pos) * 0.80)
    train_fit = train_pos[:max(1, split - args.purge)]
    val_pos = train_pos[min(len(train_pos), split + args.purge):]
    if len(train_fit) < 1000 or len(val_pos) < 200:
        raise RuntimeError("Training-only train/validation split is too small")
    labels = future[valid]

    # Select features and complexity on a chronological validation tail of the training period.
    # Held-out test days are not inspected until this grid has a winner.
    grid_results = []
    for features, depth, rounds in GRID:
        X = matrices[features][valid]
        model = build_regressor(
            X[train_fit], labels[train_fit], depth=depth, rounds=rounds,
            seed=args.seed, nthread=args.nthread,
        )
        pred = model.predict(xgb.DMatrix(X[val_pos]))
        grid_results.append({
            "features": features,
            "max_depth": depth,
            "rounds": rounds,
            "validation_rows": int(len(val_pos)),
            "validation": score_regression(pred, labels[val_pos]),
        })
    selected = max(grid_results, key=lambda row: row["validation"]["pearson_ic"])
    selected_config = {k: selected[k] for k in ("features", "max_depth", "rounds")}
    selected_matrix = matrices[selected["features"]][valid]
    # Refit on the entire training file, preserving a purge gap at its end.
    final_train = train_pos[:max(1, len(train_pos) - args.purge)]
    final_model = build_regressor(
        selected_matrix[final_train], labels[final_train],
        depth=selected["max_depth"], rounds=selected["rounds"],
        seed=args.seed, nthread=args.nthread,
    )

    day_results = []
    pooled_pred, pooled_y = [], []
    for day_idx, (start, end) in enumerate(boundaries[1:], start=8):
        positions = np.flatnonzero((raw_index >= start) & (raw_index < end))
        if len(positions) < 100:
            raise RuntimeError(f"Too few valid rows for held-out day {day_idx}: {len(positions)}")
        preds = final_model.predict(xgb.DMatrix(selected_matrix[positions]))
        metrics = score_regression(preds, labels[positions])
        day_results.append({
            "day": day_idx,
            "events": int(end - start),
            "model_rows": int(len(positions)),
            "metrics": metrics,
        })
        pooled_pred.append(preds)
        pooled_y.append(labels[positions])
    pooled_pred_array = np.concatenate(pooled_pred)
    pooled_y_array = np.concatenate(pooled_y)
    result: dict[str, Any] = {
        "experiment": "FI-2010 continuous forward-return regression",
        "dataset": {
            "name": "FI-2010 limit order book benchmark",
            "landing_page": DATASET_PAGE,
            "license": DATASET_LICENSE,
            "archive_sha256": {str(p): sha256(p) for p in paths},
            "train_events": int(train_events),
            "test_events": [int(len(f)) for f in frames[1:]],
            "total_events": int(total_events),
        },
        "method": {
            "target": f"log(mid[t+{args.horizon}]) - log(mid[t])",
            "split": "chronological 80/20 train-validation split within the training file; purge around the split",
            "test_selection": "model and features selected only on the training-file validation tail; held-out segments evaluated once after refit",
            "purge_events": int(args.purge),
            "selected_d": float(d),
            "stationarity_selection": stationarity,
            "tda_backend": topology_meta["backend"],
            "tda_clouds": int(topology_meta["n_clouds"]),
            "tda_stride": int(args.tda_stride),
            "ffd_backend": ffd_backend,
            "gpu_used": bool(gpu_used),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "grid_size": len(grid_results),
            "validation_rows": int(len(val_pos)),
            "final_train_rows": int(len(final_train)),
            "candidate_grid": [{"features": f, "max_depth": depth, "rounds": rounds} for f, depth, rounds in GRID],
        },
        "validation": {
            "selected_config": selected_config,
            "selected_metrics": selected["validation"],
            "all_candidates": grid_results,
        },
        "selected_model": selected_config,
        "heldout_days": day_results,
        "heldout_pooled": score_regression(pooled_pred_array, pooled_y_array),
        "target_resume_ic": 0.064,
        "target_status": "measured target; no result is altered or substituted",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    dashboard = Path(args.dashboard) if args.dashboard else output.with_name("fi2010_regression_sweep.png")
    make_dashboard(dashboard, grid_results, selected, day_results)
    print(json.dumps(result, indent=2))
    print(f"Dashboard: {dashboard}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_regression_sweep.json")
    parser.add_argument("--dashboard", default="results/fi2010_regression_sweep.png")
    parser.add_argument("--tda-stride", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--purge", type=int, default=50)
    parser.add_argument("--ffd-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2010)
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--tda-jobs", type=int, default=-1)
    parser.add_argument("--require-gtda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1 or args.purge < args.horizon:
        parser.error("horizon must be positive and purge must be >= horizon")
    if args.tda_stride < 1 or args.nthread < 1 or args.ffd_width < 1:
        parser.error("tda-stride, nthread, and ffd-width must be positive")
    run(args)


if __name__ == "__main__":
    main()
