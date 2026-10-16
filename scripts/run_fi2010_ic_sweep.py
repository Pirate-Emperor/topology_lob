from __future__ import annotations

"""Leakage-aware FI-2010 validation sweep.

The validation segment selects only the binary label threshold and objective.
The final held-out segments (days 8, 9 and 10) are evaluated once after that
choice. They are never used to choose a configuration.
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
    select_stationary_d_by_segment,
    split_stock_sequences,
    topology_by_segment,
)

from topology_lob import (
    fractional_diff_gpu,
    fit_focal_xgb,
    fit_logloss_xgb,
    make_point_clouds,
    microstructure_features,
    persistent_features,
    select_stationary_d,
)

RADII = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)


def forward_returns(log_mid: np.ndarray, boundaries: list[tuple[int, int]], horizon: int) -> np.ndarray:
    """Build forward returns within each segment; never cross a split boundary."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    future = np.full(len(log_mid), np.nan, dtype=float)
    for start, end in boundaries:
        if end - start <= horizon:
            continue
        future[start : end - horizon] = (
            log_mid[start + horizon : end] - log_mid[start : end - horizon]
        )
    return future


def binary_labels(future: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    y = np.full(len(future), -1, dtype=np.int8)
    valid = np.isfinite(future)
    y[valid] = (future[valid] > threshold).astype(np.int8)
    return y


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def measure(model, X: np.ndarray, y: np.ndarray, future: np.ndarray) -> dict:
    margin = model.predict(xgb.DMatrix(X), output_margin=True)
    probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "pearson_ic": corr(probability, future),
        "rank_ic": float(spearmanr(probability, future).statistic)
        if np.std(probability) and np.std(future) else 0.0,
        "margin_pearson_ic": corr(margin, future),
        "margin_rank_ic": float(spearmanr(margin, future).statistic)
        if np.std(margin) and np.std(future) else 0.0,
    }


def build_features(args):
    train = raw40_to_canonical(load_raw(args.train).features)
    tests = [raw40_to_canonical(load_raw(path).features) for path in args.test]
    frames = [train, *tests]
    combined = pd.concat(frames, ignore_index=True)
    train_events = len(train)
    sequence_frames, file_boundaries, boundaries, sequence_cuts = split_stock_sequences(
        frames,
        threshold_pct=float(getattr(args, "sequence_jump_threshold", 0.05)),
        min_segment_events=int(getattr(args, "min_segment_events", 1024)),
    )
    micro, measured_boundaries = microstructure_by_segment(sequence_frames, depth_levels=5)
    if measured_boundaries != boundaries:
        raise RuntimeError("feature helper boundaries do not match stock splits")

    d, stationarity = select_stationary_d_by_segment(
        micro["log_mid"].to_numpy(),
        boundaries,
        train_events,
        np.linspace(0.0, 1.0, 11),
        0.05,
        args.ffd_width,
        prefix_fraction=args.stationarity_fraction,
    )
    fd, gpu_used, fd_backend = fractional_diff_by_segment(
        micro["log_mid"].to_numpy(), boundaries, d, args.ffd_width,
        require_cuda=args.require_cuda,
    )
    micro["fracdiff"] = fd

    # Inter-arrival is a constant synthetic event clock in FI-2010, so it is
    # deliberately excluded from model features.
    micro_cols = [
        c for c in micro.columns
        if c not in {"mid", "log_mid", "interarrival_us", "fracdiff"}
    ]
    fd_cols = micro_cols + ["fracdiff"]
    topo_df, topo_meta, topo_indices = topology_by_segment(
        sequence_frames, boundaries, RADII,
        horizon=args.horizon, stride=args.tda_stride, levels=10,
        require_gtda=args.require_gtda, n_jobs=args.tda_jobs,
    )
    features = np.hstack([micro[fd_cols].to_numpy(float), topo_df.to_numpy(float)])
    future = forward_returns(micro["log_mid"].to_numpy(), boundaries, args.horizon)
    valid = np.isfinite(features).all(axis=1) & np.isfinite(future)
    indices = np.flatnonzero(valid)
    features = features[valid]
    future = future[valid]

    return {
        "features": features,
        "future": future,
        "indices": indices,
        "train_events": train_events,
        "test_events": [len(x) for x in tests],
        "stationarity": stationarity,
        "selected_d": float(d),
        "tda": topo_meta,
        "ffd_backend": fd_backend,
        "gpu_used": bool(gpu_used),
        "topology_clouds": int(topo_meta["n_clouds"]),
        "boundaries": boundaries,
        "file_boundaries": file_boundaries,
        "sequence_cuts": sequence_cuts,
        "sequence_frames": sequence_frames,
        "stock_sequences_per_file": [
            int(sum(1 for b in boundaries if file_start <= b[0] and b[1] <= file_end))
            for file_start, file_end in file_boundaries
        ],
    }


def fit_model(objective: str, X: np.ndarray, y: np.ndarray, seed: int):
    if objective == "focal":
        return fit_focal_xgb(X, y, seed=seed)
    if objective == "logloss":
        return fit_logloss_xgb(X, y, seed=seed)
    raise ValueError(f"Unknown objective: {objective}")


def run(args) -> dict:
    start = time.perf_counter()
    b = build_features(args)
    X = b["features"]
    indices = b["indices"]
    future = b["future"]
    train_events = b["train_events"]
    purge = args.purge
    boundaries = b["boundaries"]
    file_boundaries = b["file_boundaries"]

    fit_mask = np.zeros(len(indices), dtype=bool)
    val_mask = np.zeros(len(indices), dtype=bool)
    final_train_mask = indices < train_events
    for start, end in boundaries:
        if end <= train_events:
            split = start + int((end - start) * args.validation_start)
            fit_mask |= (indices >= start) & (indices < split - purge)
            val_mask |= (indices >= split + purge) & (indices < end - args.horizon)
            final_train_mask &= ~(
                (indices >= max(start, end - max(purge, args.horizon)))
                & (indices < end)
            )
    test_mask = indices >= train_events
    if fit_mask.sum() < 500 or val_mask.sum() < 2000 or final_train_mask.sum() < 1000 or test_mask.sum() < 1000:
        raise RuntimeError(
            f"Not enough rows for split: fit={fit_mask.sum()}, "
            f"validation={val_mask.sum()}, final_train={final_train_mask.sum()}, test={test_mask.sum()}"
        )

    candidates = []
    for threshold in args.thresholds:
        y_all = binary_labels(future, threshold)
        fit_positions = np.flatnonzero(fit_mask & (y_all >= 0))
        val_positions = np.flatnonzero(val_mask & (y_all >= 0))
        if len(np.unique(y_all[fit_positions])) < 2:
            candidates.append({
                "threshold": float(threshold), "objective": None,
                "eligible": False, "reason": "training labels contain one class",
            })
            continue
        for objective in ("focal", "logloss"):
            model = fit_model(objective, X[fit_positions], y_all[fit_positions], args.seed)
            score = measure(model, X[val_positions], y_all[val_positions], future[val_positions])
            row = {
                "threshold": float(threshold),
                "objective": objective,
                "eligible": True,
                "validation": score,
            }
            candidates.append(row)

    eligible = [r for r in candidates if r.get("eligible") and r["validation"]["rows"]]
    if not eligible:
        raise RuntimeError("No eligible configuration reached validation")
    # Select from the internal validation set only. Test days are not used to
    # select the objective or threshold.
    selected = max(
        eligible,
        key=lambda r: (r["validation"]["pearson_ic"], r["validation"]["rank_ic"]),
    )

    y_default = binary_labels(future, args.default_threshold)
    y_selected = binary_labels(future, selected["threshold"])
    configs = {
        "fixed_reference": {"threshold": float(args.default_threshold), "objective": "focal"},
        "validation_selected": {"threshold": selected["threshold"], "objective": selected["objective"]},
    }
    final_models = {}
    for name, config in configs.items():
        labels = y_default if name == "fixed_reference" else y_selected
        train_positions = np.flatnonzero(final_train_mask & (labels >= 0))
        if len(np.unique(labels[train_positions])) < 2:
            raise RuntimeError(f"Training labels contain one class for {name}")
        final_models[name] = fit_model(
            config["objective"], X[train_positions], labels[train_positions], args.seed
        )

    test_days = []
    test_boundaries = file_boundaries[1:]
    for day_idx, (start_idx, end_idx) in enumerate(test_boundaries, start=8):
        pos = np.flatnonzero((indices >= start_idx) & (indices < end_idx))
        if len(pos) == 0:
            raise RuntimeError(f"No valid test rows on segment {day_idx}")
        day = {"segment": day_idx, "events": int(end_idx - start_idx)}
        for name in configs:
            labels = y_default if name == "fixed_reference" else y_selected
            day[name] = measure(final_models[name], X[pos], labels[pos], future[pos])
        test_days.append(day)

    pooled = {}
    pooled_pos = np.flatnonzero(test_mask)
    for name in configs:
        labels = y_default if name == "fixed_reference" else y_selected
        pooled[name] = measure(
            final_models[name], X[pooled_pos], labels[pooled_pos], future[pooled_pos]
        )

    selected_test = pooled["validation_selected"]
    elapsed_seconds = float(time.perf_counter() - start)
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise RuntimeError(f"Invalid monotonic elapsed time: {elapsed_seconds!r}")
    result = {
        "data_source": "FI-2010 Decimal-Precision benchmark",
        "data_license": "CC BY 4.0; attribute Ntakaris et al. (2018)",
        "paper": "Ntakaris et al., Journal of Forecasting 37(8), 852-866 (2018), DOI 10.1002/for.2543",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "seed": args.seed,
        "train_events": int(train_events),
        "sequence_jump_threshold_pct": float(getattr(args, "sequence_jump_threshold", 0.05)) * 100.0,
        "sequence_cuts": b["sequence_cuts"],
        "stock_sequences_per_file": b["stock_sequences_per_file"],
        "test_events": [int(n) for n in b["test_events"]],
        "horizon": int(args.horizon),
        "purge_events": int(purge),
        "selected_d": b["selected_d"],
        "stationarity": b["stationarity"],
        "tda": b["tda"],
        "topology_clouds": b["topology_clouds"],
        "fractional_diff_backend": b["ffd_backend"],
        "gpu_used": b["gpu_used"],
        "internal_validation": {
            "fit_rows": int(fit_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "selection_metric": "Pearson IC on internal chronological validation",
            "candidates": candidates,
            "selected": selected,
        },
        "heldout_days": test_days,
        "heldout_pooled": pooled,
        "target_resume_ic": 0.064,
        "target_metric_status": {
            "pearson_ic_at_least_0_064": bool(selected_test["pearson_ic"] >= 0.064),
            "rank_ic_at_least_0_064": bool(selected_test["rank_ic"] >= 0.064),
            "warning": "These flags are measured on held-out data; the target value is never substituted into the results.",
        },
        "elapsed_seconds": elapsed_seconds,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_ic_sweep.json")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.0001, 0.00025, 0.0005, 0.001])
    parser.add_argument("--default-threshold", type=float, default=0.0005)
    parser.add_argument("--purge", type=int, default=50)
    parser.add_argument("--ffd-width", type=int, default=256)
    parser.add_argument("--tda-stride", type=int, default=1000)
    parser.add_argument("--tda-jobs", type=int, default=-1)
    parser.add_argument("--stationarity-fraction", type=float, default=0.6)
    parser.add_argument("--sequence-jump-threshold", type=float, default=0.05)
    parser.add_argument("--min-segment-events", type=int, default=1024)
    parser.add_argument("--validation-start", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2010)
    parser.add_argument("--require-gtda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not 0.2 <= args.stationarity_fraction < args.validation_start < 0.95:
        raise SystemExit("Require 0.2 <= stationarity-fraction < validation-start < 0.95")
    if args.purge < args.horizon:
        raise SystemExit("Purge must be at least the forward-return horizon")
    if not args.thresholds or not np.isfinite(args.thresholds).all():
        raise SystemExit("Thresholds must be a non-empty finite list")
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
