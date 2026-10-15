from __future__ import annotations

"""Fixed, no-tuning FI-2010 continuous-return regression and feature ablation.

Unlike the binary classification sweep, this experiment predicts the signed
continuous forward return directly. It evaluates every declared horizon and
feature set rather than choosing a winner from the held-out test results.
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
    topology_by_segment,
)

from topology_lob import (
    fractional_diff_gpu,
    make_point_clouds,
    microstructure_features,
    persistent_features,
    select_stationary_d,
)

RADII = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
FEATURE_SETS = ("micro_only", "micro_plus_fd", "micro_plus_topology", "full")


def forward_returns(
    log_mid: np.ndarray,
    boundaries: list[tuple[int, int]],
    horizon: int,
) -> np.ndarray:
    """Calculate continuous forward log returns without crossing file splits."""
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


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(a) != len(b):
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("correlation inputs must be finite")
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score(prediction: np.ndarray, future: np.ndarray) -> dict:
    return {
        "rows": int(len(future)),
        "pearson_ic": correlation(prediction, future),
        "rank_ic": float(spearmanr(prediction, future).statistic)
        if len(future) > 1 and np.std(prediction) and np.std(future) else 0.0,
        "direction_hit_rate": float(np.mean((prediction > 0) == (future > 0))),
        "prediction_std": float(np.std(prediction)),
        "future_return_std": float(np.std(future)),
        "mean_prediction_bps": float(np.mean(prediction) * 1e4),
        "mean_future_return_bps": float(np.mean(future) * 1e4),
    }


def read_segments(train_path: str, test_paths: list[str]):
    frames = [raw40_to_canonical(load_raw(path).features) for path in [train_path, *test_paths]]
    boundaries = []
    cursor = 0
    for frame in frames:
        boundaries.append((cursor, cursor + len(frame)))
        cursor += len(frame)
    return frames, boundaries


def build_features(args) -> dict:
    frames, boundaries = read_segments(args.train, args.test)
    combined = pd.concat(frames, ignore_index=True)
    train_events = len(frames[0])
    select_end = int(train_events * args.stationarity_fraction)
    micro, boundaries = microstructure_by_segment(frames, depth_levels=5)

    d, stationarity = select_stationary_d(
        micro["log_mid"].to_numpy()[:select_end],
        np.linspace(0.0, 1.0, 11),
        0.05,
        args.ffd_width,
    )
    fd, gpu_used, fd_backend = fractional_diff_by_segment(
        micro["log_mid"].to_numpy(), boundaries, d, args.ffd_width,
        require_cuda=args.require_cuda,
    )
    micro["fracdiff"] = fd

    micro_cols = [
        c for c in micro.columns
        if c not in {"mid", "log_mid", "interarrival_us", "fracdiff"}
    ]
    feature_arrays = {
        "micro_only": (micro[micro_cols].to_numpy(float), list(micro_cols)),
        "micro_plus_fd": (
            micro[micro_cols + ["fracdiff"]].to_numpy(float),
            micro_cols + ["fracdiff"],
        ),
    }
    topo_df, topo_meta, topo_indices = topology_by_segment(
        frames, boundaries, RADII,
        horizon=max(args.horizons), stride=args.tda_stride, levels=10,
        require_gtda=args.require_gtda, n_jobs=args.tda_jobs,
    )
    topo_names = list(topo_meta["feature_names"])
    topo_arr = topo_df.to_numpy(float)
    feature_arrays["micro_plus_topology"] = (
        np.hstack([micro[micro_cols].to_numpy(float), topo_arr]),
        micro_cols + topo_names,
    )
    feature_arrays["full"] = (
        np.hstack([micro[micro_cols + ["fracdiff"]].to_numpy(float), topo_arr]),
        micro_cols + ["fracdiff"] + topo_names,
    )
    return {
        "micro": micro,
        "frames": frames,
        "boundaries": boundaries,
        "train_events": train_events,
        "feature_arrays": feature_arrays,
        "stationarity": stationarity,
        "selected_d": float(d),
        "tda": topo_meta,
        "topology_clouds": int(topo_meta["n_clouds"]),
        "ffd_backend": fd_backend,
        "gpu_used": bool(gpu_used),
    }


def fit_regressor(X_train: np.ndarray, y_train: np.ndarray, seed: int, rounds: int, nthread: int = 4):
    # Return scale is in basis points to keep tree gradients numerically useful.
    target_bps = np.asarray(y_train, dtype=float) * 1e4
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 8,
        "lambda": 2.0,
        "seed": seed,
        "nthread": int(nthread),
    }
    return xgb.train(
        params,
        xgb.DMatrix(X_train, label=target_bps),
        num_boost_round=rounds,
        verbose_eval=False,
    )


def run(args) -> dict:
    started = time.perf_counter()
    bundle = build_features(args)
    indices = np.arange(len(bundle["micro"]))
    train_events = bundle["train_events"]
    boundaries = bundle["boundaries"]
    final_train_end = train_events - args.purge
    results = []

    for horizon in args.horizons:
        future = forward_returns(
            bundle["micro"]["log_mid"].to_numpy(), boundaries, horizon
        )
        purge = max(args.purge, horizon)
        for feature_name in FEATURE_SETS:
            X_all, feature_names = bundle["feature_arrays"][feature_name]
            valid = np.isfinite(X_all).all(axis=1) & np.isfinite(future)
            train_mask = valid & (indices < train_events - purge)
            test_mask = valid & (indices >= train_events)
            train_pos = np.flatnonzero(train_mask)
            test_pos = np.flatnonzero(test_mask)
            if len(train_pos) < 1000 or len(test_pos) < 1000:
                raise RuntimeError(
                    f"not enough rows: h={horizon}, features={feature_name}, "
                    f"train={len(train_pos)}, test={len(test_pos)}"
                )

            model = fit_regressor(
                X_all[train_pos], future[train_pos], args.seed + horizon, args.rounds
            )
            prediction = model.predict(xgb.DMatrix(X_all[test_pos])) / 1e4
            heldout = {
                "segment_metrics": [],
                "pooled": score(prediction, future[test_pos]),
            }
            for segment_no, (start, end) in enumerate(boundaries[1:], start=8):
                local = np.flatnonzero(
                    (indices[test_pos] >= start) & (indices[test_pos] < end)
                )
                heldout["segment_metrics"].append({
                    "segment": segment_no,
                    "events": int(end - start),
                    **score(prediction[local], future[test_pos[local]]),
                })
            results.append({
                "horizon": int(horizon),
                "feature_set": feature_name,
                "feature_count": int(X_all.shape[1]),
                "train_rows": int(len(train_pos)),
                "test_rows": int(len(test_pos)),
                "purge_events": int(purge),
                "heldout": heldout,
            })

    # Descriptive ranking only: no model/horizon selection is performed from
    # these test scores. All 20 settings are fully reported below.
    top_test_configs = sorted(
        [
            {
                "horizon": r["horizon"],
                "feature_set": r["feature_set"],
                "pearson_ic": r["heldout"]["pooled"]["pearson_ic"],
                "rank_ic": r["heldout"]["pooled"]["rank_ic"],
            }
            for r in results
        ],
        key=lambda r: r["pearson_ic"],
        reverse=True,
    )[:5]
    return {
        "data_source": "FI-2010 Decimal-Precision benchmark",
        "data_license": "CC BY 4.0; cite Ntakaris et al. (2018)",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "experiment": "continuous forward-return XGBoost regression; fixed configs, no test-based selection",
        "train_events": int(train_events),
        "test_events": [int(len(f)) for f in bundle["frames"][1:]],
        "horizons": list(map(int, args.horizons)),
        "rounds": int(args.rounds),
        "purge_minimum": int(args.purge),
        "selected_d": bundle["selected_d"],
        "stationarity": bundle["stationarity"],
        "tda": bundle["tda"],
        "topology_clouds": bundle["topology_clouds"],
        "ffd_backend": bundle["ffd_backend"],
        "gpu_used": bundle["gpu_used"],
        "results": results,
        "top_5_heldout_pearson_ic_descriptive_only": top_test_configs,
        "target_pearson_ic_0_064_reached_by_any_fixed_config": bool(
            any(r["heldout"]["pooled"]["pearson_ic"] >= 0.064 for r in results)
        ),
        "target_rank_ic_0_064_reached_by_any_fixed_config": bool(
            any(r["heldout"]["pooled"]["rank_ic"] >= 0.064 for r in results)
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_regression_sweep.json")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 30, 50, 100])
    parser.add_argument("--purge", type=int, default=50)
    parser.add_argument("--ffd-width", type=int, default=256)
    parser.add_argument("--tda-stride", type=int, default=1000)
    parser.add_argument("--tda-jobs", type=int, default=-1)
    parser.add_argument("--stationarity-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2010)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--require-gtda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not args.horizons or any(h < 1 for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        raise SystemExit("Horizons must be unique positive integers")
    if args.purge < 1 or args.ffd_width < 1 or args.rounds < 1:
        raise SystemExit("purge, ffd-width, and rounds must be positive")
    if not 0.2 <= args.stationarity_fraction <= 0.8:
        raise SystemExit("stationarity-fraction must be between 0.2 and 0.8")
    result = run(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
