from __future__ import annotations

import argparse
import json
import sys
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
    evaluate,
    make_point_clouds,
    microstructure_features,
    persistent_features,
    select_stationary_d,
)


def read_canonical(path: str | Path) -> pd.DataFrame:
    frame = raw40_to_canonical(load_raw(path).features)
    return frame


def evaluate_signal(model, X, y, future):
    """Evaluate both probability and raw XGBoost margin as signals."""
    matrix = xgb.DMatrix(X)
    margin = model.predict(matrix, output_margin=True)
    probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise RuntimeError("Custom-objective probability transform left [0,1]")

    base, _ = evaluate(model, X, y, future)
    base["margin_pearson_ic"] = (
        float(np.corrcoef(margin, future)[0, 1])
        if np.std(margin) and np.std(future) else 0.0
    )
    base["margin_rank_ic"] = (
        float(spearmanr(margin, future).statistic)
        if np.std(margin) and np.std(future) else 0.0
    )
    base["probability_mean"] = float(np.mean(probability))
    base["margin_mean"] = float(np.mean(margin))
    return base, probability, margin


def segment_forward_returns(log_mid: np.ndarray, boundaries: list[tuple[int, int]], horizon: int) -> np.ndarray:
    future = np.full(len(log_mid), np.nan)
    for start, end in boundaries:
        if end - start <= horizon:
            continue
        future[start:end-horizon] = (
            log_mid[start+horizon:end] - log_mid[start:end-horizon]
        )
    return future


def main():
    ap = argparse.ArgumentParser(
        description="Leakage-aware FI-2010 train-7/test-8,9,10 benchmark"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True, nargs=3)
    ap.add_argument("--out", default="results/fi2010_benchmark.json")
    ap.add_argument("--tda-stride", type=int, default=500)
    ap.add_argument("--require-gtda", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--label-threshold", type=float, default=0.0005)
    ap.add_argument("--purge", type=int, default=50)
    ap.add_argument("--ffd-width", type=int, default=256)
    ap.add_argument("--sequence-jump-threshold", type=float, default=0.05)
    ap.add_argument("--min-segment-events", type=int, default=1024)
    args = ap.parse_args()

    train = read_canonical(args.train)
    tests = [read_canonical(path) for path in args.test]
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

    # Use only training-stock segments to pick d; never run ADF over different stocks back-to-back.
    d, stationarity = select_stationary_d_by_segment(
        micro["log_mid"].to_numpy(),
        boundaries,
        train_events,
        np.linspace(0.0, 1.0, 11),
        0.05,
        args.ffd_width,
        prefix_fraction=1.0,
    )
    fd, gpu_used, fd_backend = fractional_diff_by_segment(
        micro["log_mid"].to_numpy(), boundaries, d, args.ffd_width,
        require_cuda=args.require_cuda,
    )
    micro["fracdiff"] = fd
    radii = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    topo_df, topo_meta, topo_index = topology_by_segment(
        sequence_frames, boundaries, radii,
        horizon=args.horizon, stride=args.tda_stride, levels=10,
        require_gtda=args.require_gtda, n_jobs=-1,
    )

    base_columns = [
        c for c in micro.columns
        if c not in {"mid", "log_mid"}
    ]
    X_all = np.hstack([
        micro[base_columns].to_numpy(float),
        topo_df.to_numpy(float),
    ])

    mid_log = micro["log_mid"].to_numpy()
    future = segment_forward_returns(mid_log, boundaries, args.horizon)
    y = np.full(len(combined), -1)
    finite = np.isfinite(future)
    y[finite] = (future[finite] > args.label_threshold).astype(int)

    y = np.full(len(combined), -1)
    finite = np.isfinite(future)
    y[finite] = (
        future[finite] > args.label_threshold
    ).astype(int)

    valid = (
        np.isfinite(X_all).all(axis=1)
        & np.isfinite(future)
        & (y >= 0)
    )

    X = X_all[valid]
    yy = y[valid]
    rr = future[valid]
    raw_indices = np.flatnonzero(valid)

    train_mask = raw_indices < train_events
    test_mask = raw_indices >= train_events

    # Purge the end of the training stream so labels/windows near the
    # train/test boundary cannot overlap the test interval.
    train_positions = np.flatnonzero(train_mask)
    test_positions = np.flatnonzero(test_mask)
    purge = max(args.purge, args.horizon)
    train_mask = train_mask.copy()
    for start, end in boundaries:
        if end <= train_events:
            train_mask &= ~(
                (raw_indices >= max(start, end - purge)) & (raw_indices < end)
            )
    train_positions = np.flatnonzero(train_mask)

    X_train = X[train_positions]
    y_train = yy[train_positions]
    results = {
        "data_source": "FI-2010 Decimal-Precision benchmark",
        "data_license": "CC BY 4.0; cite Ntakaris et al. (2018)",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "sequence_jump_threshold_pct": float(getattr(args, "sequence_jump_threshold", 0.05)) * 100.0,
        "sequence_cuts": sequence_cuts,
        "stock_sequences_per_file": [
            int(sum(1 for b in boundaries if file_start <= b[0] and b[1] <= file_end))
            for file_start, file_end in file_boundaries
        ],
        "train_events": int(train_events),
        "test_events": [int(len(x)) for x in tests],
        "combined_events": int(len(combined)),
        "selected_d": float(d),
        "stationarity": stationarity,
        "tda": topo_meta,
        "fractional_diff_backend": fd_backend,
        "gpu_used": bool(gpu_used),
        "purge_events_per_stock_sequence": int(purge),
        "label_threshold": float(args.label_threshold),
        "horizon": int(args.horizon),
        "target_resume_ic": 0.064,
        "target_metric_status": "measured on held-out days; Pearson IC and Rank IC reported separately",
        "days": [],
    }

    focal = fit_focal_xgb(X_train, y_train, seed=2010)
    baseline = fit_logloss_xgb(X_train, y_train, seed=2010)

    pooled_probability = []
    pooled_margin = []
    pooled_future = []
    cursor = train_events
    for day_idx, test_frame in enumerate(tests, start=8):
        end = cursor + len(test_frame)
        day_positions = np.flatnonzero(
            (raw_indices >= cursor) & (raw_indices < end)
        )
        if not len(day_positions):
            raise RuntimeError(f"No valid model rows remained for test day {day_idx}")
        focal_metrics, focal_probability, focal_margin = evaluate_signal(
            focal,
            X[day_positions],
            yy[day_positions],
            rr[day_positions],
        )
        baseline_metrics, _, _ = evaluate_signal(
            baseline,
            X[day_positions],
            yy[day_positions],
            rr[day_positions],
        )
        pooled_probability.append(focal_probability)
        pooled_margin.append(focal_margin)
        pooled_future.append(rr[day_positions])
        results["days"].append({
            "day": day_idx,
            "events": int(len(test_frame)),
            "model_rows": int(len(day_positions)),
            "focal": focal_metrics,
            "logloss": baseline_metrics,
        })
        cursor = end

    ics = [d["focal"]["pearson_ic"] for d in results["days"]]
    rank_ics = [d["focal"]["rank_ic"] for d in results["days"]]
    margin_ics = [d["focal"]["margin_pearson_ic"] for d in results["days"]]
    margin_rank_ics = [d["focal"]["margin_rank_ic"] for d in results["days"]]
    pooled_probability = np.concatenate(pooled_probability)
    pooled_margin = np.concatenate(pooled_margin)
    pooled_future = np.concatenate(pooled_future)
    results["aggregate"] = {
        "focal_mean_pearson_ic": float(np.mean(ics)),
        "focal_std_pearson_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
        "focal_mean_rank_ic": float(np.mean(rank_ics)),
        "focal_std_rank_ic": float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
        "focal_mean_margin_pearson_ic": float(np.mean(margin_ics)),
        "focal_std_margin_pearson_ic": float(np.std(margin_ics, ddof=1)) if len(margin_ics) > 1 else 0.0,
        "focal_mean_margin_rank_ic": float(np.mean(margin_rank_ics)),
        "focal_std_margin_rank_ic": float(np.std(margin_rank_ics, ddof=1)) if len(margin_rank_ics) > 1 else 0.0,
        "focal_pooled_probability_pearson_ic": float(np.corrcoef(pooled_probability, pooled_future)[0, 1]),
        "focal_pooled_margin_pearson_ic": float(np.corrcoef(pooled_margin, pooled_future)[0, 1]),
        "focal_pooled_margin_rank_ic": float(spearmanr(pooled_margin, pooled_future).statistic),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
