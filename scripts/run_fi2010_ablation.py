from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_model(model, X, y, future):
    dm = xgb.DMatrix(X)
    margin = model.predict(dm, output_margin=True)
    prob = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
    roc = None
    if np.unique(y).size == 2:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, prob))
    return {
        "probability_pearson_ic": corr(prob, future),
        "margin_pearson_ic": corr(margin, future),
        "rank_ic": float(spearmanr(margin, future).statistic),
        "roc_auc": roc,
    }


def prepare(args):
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
        prefix_fraction=1.0,
    )
    fd, gpu_used, fd_backend = fractional_diff_by_segment(
        micro["log_mid"].to_numpy(), boundaries, d, args.ffd_width,
        require_cuda=args.require_cuda,
    )
    micro["fracdiff"] = fd
    base = [
        c for c in micro.columns
        if c not in {"mid", "log_mid", "interarrival_us"}
    ]
    radii = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5])
    topo_df, topo_meta, topo_index = topology_by_segment(
        sequence_frames, boundaries, radii,
        horizon=args.horizon, stride=args.tda_stride, levels=10,
        require_gtda=args.require_gtda, n_jobs=-1,
    )

    logs = micro["log_mid"].to_numpy()
    future = segment_forward_returns(logs, boundaries, args.horizon)
    y = np.full(len(combined), -1)
    finite = np.isfinite(future)
    y[finite] = (future[finite] > args.label_threshold).astype(int)

    y = np.full(len(combined), -1)
    finite = np.isfinite(future)
    y[finite] = (future[finite] > args.label_threshold).astype(int)

    arrays = {}
    micro_cols = [c for c in base if c != "fracdiff"]
    arrays["micro_only"] = micro[micro_cols].to_numpy(float)
    arrays["micro_plus_fd"] = micro[base].to_numpy(float)
    arrays["micro_plus_topology"] = np.hstack([
        micro[micro_cols].to_numpy(float),
        topo_df.to_numpy(float),
    ])
    arrays["full"] = np.hstack([
        micro[base].to_numpy(float),
        topo_df.to_numpy(float),
    ])

    results = {
        "data_source": "FI-2010 Decimal-Precision",
        "train_events": train_events,
        "test_events": [len(x) for x in tests],
        "data_license": "CC BY 4.0; cite Ntakaris et al. (2018)",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "sequence_jump_threshold_pct": float(getattr(args, "sequence_jump_threshold", 0.05)) * 100.0,
        "sequence_cuts": sequence_cuts,
        "stock_sequences_per_file": [
            int(sum(1 for b in boundaries if file_start <= b[0] and b[1] <= file_end))
            for file_start, file_end in file_boundaries
        ],
        "selected_d": float(d),
        "stationarity": stationarity,
        "tda": topo_meta,
        "fractional_diff_backend": fd_backend,
        "gpu_used": bool(gpu_used),
        "experiments": {},
    }

    valid = None
    for mat in arrays.values():
        current = np.isfinite(mat).all(axis=1) & np.isfinite(future) & (y >= 0)
        valid = current if valid is None else valid & current

    event_indices = np.flatnonzero(valid)
    train_mask = event_indices < train_events
    for start, end in boundaries:
        if end <= train_events:
            train_mask &= ~(
                (event_indices >= max(start, end - max(args.purge, args.horizon)))
                & (event_indices < end)
            )
    train_pos = np.flatnonzero(train_mask)

    cursor = train_events
    test_positions_by_day = []
    for test_frame in tests:
        end = cursor + len(test_frame)
        test_positions_by_day.append(np.flatnonzero(
            (event_indices >= cursor) & (event_indices < end)
        ))
        cursor = end

    pooled_test = np.concatenate(test_positions_by_day)

    for name, mat in arrays.items():
        X = mat[valid]
        yv = y[valid]
        fv = future[valid]
        focal = fit_focal_xgb(X[train_pos], yv[train_pos], seed=2010)
        baseline = fit_logloss_xgb(X[train_pos], yv[train_pos], seed=2010)

        days = []
        for day, pos in enumerate(test_positions_by_day, start=8):
            days.append({
                "day": day,
                "focal": evaluate_model(focal, X[pos], yv[pos], fv[pos]),
                "logloss": evaluate_model(baseline, X[pos], yv[pos], fv[pos]),
            })

        results["experiments"][name] = {
            "days": days,
            "pooled_focal": evaluate_model(
                focal, X[pooled_test], yv[pooled_test], fv[pooled_test]
            ),
            "pooled_logloss": evaluate_model(
                baseline, X[pooled_test], yv[pooled_test], fv[pooled_test]
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs=3, required=True)
    ap.add_argument("--out", default="results/fi2010_ablation.json")
    ap.add_argument("--tda-stride", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--label-threshold", type=float, default=0.0005)
    ap.add_argument("--purge", type=int, default=50)
    ap.add_argument("--ffd-width", type=int, default=256)
    ap.add_argument("--sequence-jump-threshold", type=float, default=0.05)
    ap.add_argument("--min-segment-events", type=int, default=1024)
    ap.add_argument("--require-gtda", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
