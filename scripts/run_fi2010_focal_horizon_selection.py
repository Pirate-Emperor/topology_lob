from __future__ import annotations

"""Training-only selection of a Focal Loss return-direction model.

All choices (forecast horizon and feature set) are selected from a chronological
validation slice of the training segment. The held-out FI-2010 days are scored
only after the candidate is fixed. The target IC is never used for selection.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topology_lob import evaluate, fit_focal_xgb, fit_logloss_xgb
from scripts.run_fi2010_return_regression import (
    FEATURE_SETS,
    build_features,
    forward_returns,
)

HORIZONS = (10, 20, 30, 50, 100, 200, 500, 1000)
LABEL_THRESHOLD = 0.0


def direction_labels(future: np.ndarray, threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    future = np.asarray(future, dtype=float)
    if future.ndim != 1 or not np.isfinite(threshold):
        raise ValueError("future must be 1-D and threshold finite")
    labels = np.full(len(future), -1, dtype=np.int8)
    valid = np.isfinite(future)
    labels[valid] = (future[valid] > threshold).astype(np.int8)
    return labels


def validate_candidate(candidate: dict) -> None:
    required = {"horizon", "feature_set", "validation"}
    if not required.issubset(candidate):
        raise ValueError(f"candidate is missing keys: {sorted(required - candidate.keys())}")
    if candidate["horizon"] not in HORIZONS:
        raise ValueError("unsupported horizon")
    if candidate["feature_set"] not in FEATURE_SETS:
        raise ValueError("unsupported feature set")


def split_positions(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    train_events: int,
    horizon: int,
    purge: int,
    fit_end: int | None = None,
    validation_start: int | None = None,
    fit_fraction: float | None = None,
    validation_fraction: float | None = None,
    sequence_boundaries: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite fit/validation positions, with optional per-stock splits.

    The fit_end/validation_start arguments preserve the original single-stream
    API. New benchmark code supplies fractions and sequence boundaries so
    each training stock is split independently.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    idx = np.asarray(indices, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or idx.ndim != 1:
        raise ValueError("features must be 2-D and labels/indices must be 1-D")
    if len(X) != len(y) or len(y) != len(idx):
        raise ValueError("features, labels, and indices must have matching lengths")
    row_valid = (y >= 0) & np.isfinite(X).all(axis=1)
    gap = max(purge, horizon)
    fit = np.zeros(len(idx), dtype=bool)
    validation = np.zeros(len(idx), dtype=bool)

    if sequence_boundaries is None:
        # Original API: exact absolute cut points inside one stream.
        if fit_end is None or validation_start is None:
            raise ValueError("provide fit_end/validation_start or stock fractions/boundaries")
        fit = row_valid & (idx < fit_end - gap)
        validation = row_valid & (idx >= validation_start + gap) & (idx < train_events - horizon)
        return np.flatnonzero(fit), np.flatnonzero(validation)

    if fit_fraction is None or validation_fraction is None:
        raise ValueError("per-stock splitting requires fit_fraction and validation_fraction")
    if not 0 < fit_fraction < validation_fraction < 1:
        raise ValueError("require 0 < fit_fraction < validation_fraction < 1")
    used = 0
    for start, end in sequence_boundaries:
        if start >= train_events:
            continue
        if start < 0 or end > train_events or start >= end:
            raise ValueError(f"invalid training stock sequence: {(start, end)}")
        length = end - start
        local_fit_end = start + int(length * fit_fraction)
        local_validation_start = start + int(length * validation_fraction)
        fit |= row_valid & (idx >= start) & (idx < local_fit_end - gap)
        validation |= row_valid & (idx >= local_validation_start + gap) & (idx < end - horizon)
        used += 1
    if used == 0:
        raise ValueError("no training stock sequences were available")
    return np.flatnonzero(fit), np.flatnonzero(validation)

def run(args) -> dict:
    started = time.perf_counter()
    bundle = build_features(args)
    raw_events = len(bundle["micro"])
    train_events = bundle["train_events"]
    indices = np.arange(raw_events)
    future_log_mid = bundle["micro"]["log_mid"].to_numpy(dtype=float)
    boundaries = bundle["boundaries"]
    file_boundaries = bundle["file_boundaries"]
    X_sets = bundle["feature_arrays"]
    candidate_rows = []

    for horizon in args.horizons:
        future = forward_returns(future_log_mid, boundaries, horizon)
        labels = direction_labels(future, args.label_threshold)
        purge = max(args.purge, horizon)

        for feature_index, feature_set in enumerate(FEATURE_SETS):
            X_all, feature_names = X_sets[feature_set]
            fit_positions, validation_positions = split_positions(
                X_all,
                labels,
                indices,
                fit_fraction=args.fit_fraction,
                validation_fraction=args.validation_fraction,
                train_events=train_events,
                horizon=horizon,
                purge=purge,
                sequence_boundaries=boundaries,
            )
            if len(fit_positions) < args.minimum_fit_rows or len(validation_positions) < args.minimum_validation_rows:
                raise RuntimeError(
                    f"not enough inner split rows for horizon={horizon}, feature_set={feature_set}: "
                    f"fit={len(fit_positions)}, validation={len(validation_positions)}"
                )
            if len(np.unique(labels[fit_positions])) != 2:
                raise RuntimeError(
                    f"single-class inner training labels for horizon={horizon}, feature_set={feature_set}"
                )

            model = fit_focal_xgb(
                X_all[fit_positions],
                labels[fit_positions],
                seed=args.seed + horizon + feature_index,
                alpha=args.focal_alpha,
                gamma=args.focal_gamma,
                ratio=args.oversample_ratio,
            )
            metrics, _ = evaluate(
                model,
                X_all[validation_positions],
                labels[validation_positions],
                future[validation_positions],
            )
            candidate_rows.append({
                "horizon": int(horizon),
                "feature_set": feature_set,
                "feature_count": int(X_all.shape[1]),
                "fit_rows": int(len(fit_positions)),
                "validation_rows": int(len(validation_positions)),
                "purge_events": int(purge),
                "validation": metrics,
            })

    # Only training-prefix validation IC determines the selected config.
    selected_row = max(
        candidate_rows,
        key=lambda r: (
            r["validation"]["pearson_ic"],
            r["validation"]["rank_ic"],
        ),
    )
    validate_candidate(selected_row)
    selected_horizon = selected_row["horizon"]
    selected_feature_set = selected_row["feature_set"]
    future = forward_returns(future_log_mid, boundaries, selected_horizon)
    labels = direction_labels(future, args.label_threshold)
    X_all, feature_names = X_sets[selected_feature_set]
    selected_purge = max(args.purge, selected_horizon)
    valid = np.isfinite(X_all).all(axis=1) & np.isfinite(future) & (labels >= 0)
    final_train_mask = valid & (indices < train_events)
    for seq_start, seq_end in boundaries:
        if seq_end <= train_events:
            final_train_mask[max(seq_start, seq_end - selected_purge):seq_end] = False
    final_train_positions = np.flatnonzero(final_train_mask)
    test_positions = np.flatnonzero(valid & (indices >= train_events))
    if len(final_train_positions) < args.minimum_fit_rows or len(test_positions) < args.minimum_test_rows:
        raise RuntimeError(
            f"not enough final split rows: train={len(final_train_positions)}, "
            f"test={len(test_positions)}"
        )
    if len(np.unique(labels[final_train_positions])) != 2:
        raise RuntimeError("single-class final training labels")

    focal = fit_focal_xgb(
        X_all[final_train_positions],
        labels[final_train_positions],
        seed=args.seed,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        ratio=args.oversample_ratio,
    )
    control = fit_logloss_xgb(
        X_all[final_train_positions],
        labels[final_train_positions],
        seed=args.seed,
        ratio=args.oversample_ratio,
    )

    focal_metrics, focal_probability = evaluate(
        focal, X_all[test_positions], labels[test_positions], future[test_positions]
    )
    control_metrics, control_probability = evaluate(
        control, X_all[test_positions], labels[test_positions], future[test_positions]
    )
    heldout_days = []
    for segment, (start, end) in enumerate(file_boundaries[1:], start=8):
        local = np.flatnonzero((indices[test_positions] >= start) & (indices[test_positions] < end))
        if len(local) == 0:
            continue
        focal_day, _ = evaluate(
            focal, X_all[test_positions[local]], labels[test_positions[local]], future[test_positions[local]]
        )
        control_day, _ = evaluate(
            control, X_all[test_positions[local]], labels[test_positions[local]], future[test_positions[local]]
        )
        heldout_days.append({
            "segment": int(segment),
            "events": int(end - start),
            "rows": int(len(local)),
            "focal": focal_day,
            "logloss_control": control_day,
        })

    selected_focal = focal_metrics
    return {
        "data_source": "FI-2010 Decimal-Precision benchmark",
        "data_license": "CC BY 4.0; cite Ntakaris et al. (2018)",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "experiment": "Focal Loss classification; prediction probability IC against continuous forward log returns",
        "selection_policy": "Select horizon and feature set using chronological validation rows from training only; score held-out segments after selection",
        "label_definition": "positive continuous forward log return; threshold 0.0; labels never cross train/test or segment boundaries",
        "target_resume_ic": 0.064,
        "fit_fraction": args.fit_fraction,
        "validation_fraction": args.validation_fraction,
        "fit_fraction_per_stock": float(args.fit_fraction),
        "validation_fraction_per_stock": float(args.validation_fraction),
        "sequence_jump_threshold_pct": 5.0,
        "sequence_cuts": bundle["sequence_cut_records"],
        "stock_sequences_per_file": [
            int(sum(1 for b in boundaries if file_start <= b[0] and b[1] <= file_end))
            for file_start, file_end in file_boundaries
        ],
        "train_events": int(train_events),
        "test_events": [int(len(f)) for f in bundle["frames"][1:]],
        "candidate_count": int(len(candidate_rows)),
        "candidate_grid": candidate_rows,
        "selected": {
            "horizon": selected_horizon,
            "feature_set": selected_feature_set,
            "feature_count": int(X_all.shape[1]),
            "fit_rows_inner": int(selected_row["fit_rows"]),
            "validation_rows": int(selected_row["validation_rows"]),
            "validation": selected_row["validation"],
            "final_train_rows": int(len(final_train_positions)),
            "test_rows": int(len(test_positions)),
            "purge_events": int(selected_purge),
        },
        "selected_test": {
            "focal": focal_metrics,
            "logloss_control": control_metrics,
            "segments": heldout_days,
        },
        "selected_test_target_status": {
            "focal_pearson_ic_at_least_0_064": bool(selected_focal["pearson_ic"] >= 0.064),
            "focal_rank_ic_at_least_0_064": bool(selected_focal["rank_ic"] >= 0.064),
            "control_pearson_ic_at_least_0_064": bool(control_metrics["pearson_ic"] >= 0.064),
            "warning": "These flags use the held-out scores. The 0.064 target is not used during selection.",
        },
        "selected_d": bundle["selected_d"],
        "stationarity": bundle["stationarity"],
        "tda": bundle["tda"],
        "topology_clouds": bundle["topology_clouds"],
        "ffd_backend": bundle["ffd_backend"],
        "gpu_used": bundle["gpu_used"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_focal_horizon_selection.json")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--purge", type=int, default=50)
    parser.add_argument("--ffd-width", type=int, default=256)
    parser.add_argument("--tda-stride", type=int, default=1000)
    parser.add_argument("--tda-jobs", type=int, default=-1)
    parser.add_argument("--stationarity-fraction", type=float, default=0.6)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-fit-rows", type=int, default=1000)
    parser.add_argument("--minimum-validation-rows", type=int, default=1000)
    parser.add_argument("--minimum-test-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2010)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--oversample-ratio", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.0)
    parser.add_argument("--require-gtda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not args.horizons or any(h not in HORIZONS for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        raise SystemExit(f"horizons must be unique values drawn from {HORIZONS}")
    if args.purge < 1 or args.ffd_width < 1 or args.minimum_fit_rows < 1:
        raise SystemExit("purge, ffd-width, and minimum rows must be positive")
    if not 0.2 <= args.stationarity_fraction <= 0.8:
        raise SystemExit("stationarity-fraction must be between 0.2 and 0.8")
    if not 0.3 <= args.fit_fraction < args.validation_fraction < 0.9:
        raise SystemExit("require 0.3 <= fit-fraction < validation-fraction < 0.9")
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
