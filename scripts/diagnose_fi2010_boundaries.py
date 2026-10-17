from __future__ import annotations

"""Diagnose possible instrument-boundary jumps in FI-2010 DecPre files.

FI-2010 files are preprocessed and do not include a reliable ticker ID or
exchange timestamps. This report does not silently alter the benchmark. It
measures large adjacent mid-price jumps and shows how much forward-return
statistics change when only fixed, predeclared discontinuity cuts are used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fi2010 import load_raw, raw40_to_canonical

JUMP_THRESHOLDS_PCT = (0.005, 0.01, 0.02, 0.05, 0.10)
HORIZONS = (10, 100, 500, 1000)


def mid_log_price(frame) -> np.ndarray:
    mid = (frame["bid_price_1"].to_numpy(float) + frame["ask_price_1"].to_numpy(float)) / 2
    if not np.isfinite(mid).all() or np.any(mid <= 0):
        raise ValueError("mid prices must be finite and positive")
    return np.log(mid)


def forward_returns(log_mid: np.ndarray, boundaries: list[tuple[int, int]], horizon: int) -> np.ndarray:
    result = np.full(len(log_mid), np.nan, dtype=float)
    for start, end in boundaries:
        if end - start > horizon:
            result[start:end - horizon] = log_mid[start + horizon:end] - log_mid[start:end - horizon]
    return result


def discontinuity_cuts(
    frame,
    threshold_pct: float = 0.02,
    min_segment_events: int = 1024,
) -> tuple[list[int], list[dict]]:
    if not 0 < threshold_pct < 1:
        raise ValueError("threshold_pct must be between 0 and 1")
    if min_segment_events < 1:
        raise ValueError("min_segment_events must be positive")
    prices = np.exp(mid_log_price(frame))
    changes = np.diff(np.log(prices))
    candidates = np.flatnonzero(np.abs(changes) >= np.log1p(threshold_pct)) + 1
    ranked = sorted(candidates.tolist(), key=lambda i: abs(changes[i - 1]), reverse=True)
    accepted: list[int] = []
    n = len(frame)
    for cut in ranked:
        if cut < min_segment_events or n - cut < min_segment_events:
            continue
        if any(abs(cut - old) < min_segment_events for old in accepted):
            continue
        accepted.append(cut)
    accepted.sort()
    detail = [
        {
            "row": int(cut),
            "log_return": float(changes[cut - 1]),
            "absolute_return_pct": float(abs(np.expm1(changes[cut - 1])) * 100),
            "previous_mid": float(prices[cut - 1]),
            "next_mid": float(prices[cut]),
        }
        for cut in accepted
    ]
    return accepted, detail


def return_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"rows": 0}
    absolute = np.abs(values)
    return {
        "rows": int(len(values)),
        "mean_bps": float(np.mean(values) * 1e4),
        "std_bps": float(np.std(values) * 1e4),
        "q50_abs_bps": float(np.quantile(absolute, 0.50) * 1e4),
        "q95_abs_bps": float(np.quantile(absolute, 0.95) * 1e4),
        "q99_abs_bps": float(np.quantile(absolute, 0.99) * 1e4),
        "q999_abs_bps": float(np.quantile(absolute, 0.999) * 1e4),
        "count_abs_ge_1pct": int(np.sum(absolute >= np.log1p(0.01))),
        "count_abs_ge_5pct": int(np.sum(absolute >= np.log1p(0.05))),
        "count_abs_ge_10pct": int(np.sum(absolute >= np.log1p(0.10))),
    }


def analyze_file(path: str, min_segment_events: int = 1024) -> dict:
    canonical = raw40_to_canonical(load_raw(path).features)
    log_mid = mid_log_price(canonical)
    delta = np.diff(log_mid)
    cuts_by_threshold = {}
    for threshold in JUMP_THRESHOLDS_PCT:
        cut_rows, details = discontinuity_cuts(canonical, threshold, min_segment_events)
        cuts_by_threshold[f"{threshold:.3f}"] = {
            "cut_count": len(cut_rows),
            "cuts": details,
            "segment_lengths": np.diff([0, *cut_rows, len(canonical)]).astype(int).tolist(),
        }
    chosen_cuts, chosen_details = discontinuity_cuts(canonical, 0.02, min_segment_events)
    boundary_arrays = [(0, len(canonical))]
    safe_bounds = list(zip([0, *chosen_cuts], [*chosen_cuts, len(canonical)]))
    horizons = {}
    for horizon in HORIZONS:
        full = forward_returns(log_mid, boundary_arrays, horizon)
        cut = forward_returns(log_mid, safe_bounds, horizon)
        horizons[str(horizon)] = {
            "within_file_only": return_summary(full),
            "after_2pct_discontinuity_cuts": return_summary(cut),
            "removed_by_cuts": int(np.isfinite(full).sum() - np.isfinite(cut).sum()),
        }
    abs_pct = np.abs(np.expm1(delta)) * 100
    top_indices = np.argsort(abs_pct)[-20:][::-1]
    return {
        "file": Path(path).name,
        "events": int(len(canonical)),
        "price_range": {
            "mid_min": float(np.min(np.exp(log_mid))),
            "mid_median": float(np.median(np.exp(log_mid))),
            "mid_max": float(np.max(np.exp(log_mid))),
        },
        "adjacent_log_return_bps": {
            "median_abs": float(np.quantile(np.abs(delta), 0.50) * 1e4),
            "p99_abs": float(np.quantile(np.abs(delta), 0.99) * 1e4),
            "p999_abs": float(np.quantile(np.abs(delta), 0.999) * 1e4),
            "max_abs": float(np.max(np.abs(delta)) * 1e4),
        },
        "threshold_counts": {
            f"{pct:.3f}": int(np.sum(np.abs(delta) >= np.log1p(pct)))
            for pct in JUMP_THRESHOLDS_PCT
        },
        "candidate_cut_sweep": cuts_by_threshold,
        "fixed_2pct_cuts": chosen_details,
        "forward_return_sensitivity": horizons,
        "largest_adjacent_jumps": [
            {
                "row": int(i + 1),
                "absolute_return_pct": float(abs_pct[i]),
                "log_return_bps": float(delta[i] * 1e4),
                "previous_mid": float(np.exp(log_mid[i])),
                "next_mid": float(np.exp(log_mid[i + 1])),
            }
            for i in top_indices
        ],
        "note": (
            "Large price jumps are candidate sequence boundaries, not proof of ticker identity. "
            "Use the predeclared 2% threshold as a diagnostic only; validate it against the source layout "
            "before using cuts in scored model runs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs=3, required=True)
    parser.add_argument("--out", default="results/fi2010_boundary_diagnostics.json")
    parser.add_argument("--min-segment-events", type=int, default=1024)
    args = parser.parse_args()
    files = [args.train, *args.test]
    report = {
        "dataset": "FI-2010 Decimal-Precision",
        "dataset_record": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "diagnostic_only": True,
        "cut_rule": {"relative_mid_jump_threshold": 0.02, "minimum_segment_events": args.min_segment_events},
        "files": [analyze_file(path, args.min_segment_events) for path in files],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    for item in report["files"]:
        print(json.dumps({
            "file": item["file"],
            "events": item["events"],
            "price_range": item["price_range"],
            "adjacent_log_return_bps": item["adjacent_log_return_bps"],
            "threshold_counts": item["threshold_counts"],
            "fixed_2pct_cuts": item["fixed_2pct_cuts"],
            "forward_return_sensitivity": item["forward_return_sensitivity"],
        }, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
