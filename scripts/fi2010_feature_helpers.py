from __future__ import annotations

"""Segment-safe feature building for FI-2010 input files."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from topology_lob import (
    fractional_diff_gpu,
    make_point_clouds,
    microstructure_features,
    persistent_features,
)


def segment_boundaries(frames: Sequence[pd.DataFrame]) -> list[tuple[int, int]]:
    if not frames or any(len(frame) == 0 for frame in frames):
        raise ValueError("segments must be non-empty")
    result = []
    cursor = 0
    for frame in frames:
        result.append((cursor, cursor + len(frame)))
        cursor += len(frame)
    return result



def split_stock_sequences(
    frames: Sequence[pd.DataFrame],
    *,
    threshold_pct: float = 0.05,
    min_segment_events: int = 1024,
) -> tuple[list[pd.DataFrame], list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    """Split FI-2010's concatenated stock streams at large price-scale jumps.

    FI-2010's public 40-column files omit security IDs. The fixed 5% rule is
    supported by the benchmark diagnostic: each input file has four such
    jumps, consistent with five stocks concatenated into one file. This is a
    predeclared, input-only rule; it never reads labels or forward returns.
    """
    if not np.isfinite(threshold_pct) or not 0 < threshold_pct < 1:
        raise ValueError("threshold_pct must be between 0 and 1")
    if min_segment_events < 1:
        raise ValueError("min_segment_events must be positive")
    file_bounds = segment_boundaries(frames)
    sequence_frames: list[pd.DataFrame] = []
    sequence_bounds: list[tuple[int, int]] = []
    cut_records: list[dict] = []
    global_offset = 0
    jump_limit = float(np.log1p(threshold_pct))

    for file_index, frame in enumerate(frames):
        if "bid_price_1" not in frame or "ask_price_1" not in frame:
            raise ValueError("FI-2010 frame is missing best bid/ask price")
        mid = (
            frame["bid_price_1"].to_numpy(dtype=float)
            + frame["ask_price_1"].to_numpy(dtype=float)
        ) / 2.0
        if not np.isfinite(mid).all() or np.any(mid <= 0):
            raise ValueError("FI-2010 mid prices must be finite and positive")
        log_mid = np.log(mid)
        jumps = np.diff(log_mid)
        candidates = (np.flatnonzero(np.abs(jumps) >= jump_limit) + 1).tolist()

        # Keep chronological cuts; adjacent detections within the minimum
        # segment size are treated as one change, not many tiny sequences.
        cuts: list[int] = []
        last_cut = 0
        for cut in candidates:
            if cut - last_cut < min_segment_events:
                continue
            if len(frame) - cut < min_segment_events:
                continue
            cuts.append(int(cut))
            last_cut = int(cut)

        edges = [0, *cuts, len(frame)]
        for part_no, (local_start, local_end) in enumerate(zip(edges[:-1], edges[1:])):
            piece = frame.iloc[local_start:local_end].reset_index(drop=True)
            if len(piece) < min_segment_events:
                raise ValueError(
                    f"price-jump split created a short sequence: file={file_index}, "
                    f"rows={local_start}:{local_end}, n={len(piece)}"
                )
            sequence_frames.append(piece)
            sequence_bounds.append((global_offset + local_start, global_offset + local_end))
            if local_start:
                cut_row = local_start
                log_return = float(jumps[cut_row - 1])
                cut_records.append({
                    "file_index": int(file_index),
                    "file_row": int(cut_row),
                    "global_row": int(global_offset + cut_row),
                    "log_return": log_return,
                    "absolute_return_pct": float(abs(np.expm1(log_return)) * 100.0),
                })
        global_offset += len(frame)

    if sum(map(len, sequence_frames)) != sum(map(len, frames)):
        raise RuntimeError("sequence split changed the number of input rows")
    if not sequence_bounds or sequence_bounds[0][0] != 0:
        raise RuntimeError("sequence boundaries do not begin at zero")
    if sequence_bounds[-1][1] != global_offset:
        raise RuntimeError("sequence boundaries do not cover all input rows")
    for left, right in zip(sequence_bounds, sequence_bounds[1:]):
        if left[1] != right[0]:
            raise RuntimeError("sequence boundaries are not contiguous")
    return sequence_frames, file_bounds, sequence_bounds, cut_records


def select_stationary_d_by_segment(
    log_mid: np.ndarray,
    sequence_boundaries: Sequence[tuple[int, int]],
    train_events: int,
    candidates: Sequence[float],
    adf_p: float = 0.05,
    width: int = 256,
    prefix_fraction: float = 0.60,
) -> tuple[float, dict]:
    """Choose one FFD order from training-only prefixes of each stock stream.

    ADF statistics are computed independently per stock segment, then combined
    by their median. This avoids treating jumps between different stocks as
    part of one price time series.
    """
    from statsmodels.tsa.stattools import adfuller
    from topology_lob import fractional_diff_cpu

    values = np.asarray(log_mid, dtype=float)
    candidate_values = np.asarray(list(candidates), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("log_mid must be a finite 1-D array")
    if train_events < 1 or train_events > len(values):
        raise ValueError("train_events is outside the input series")
    if not np.isfinite(prefix_fraction) or not 0 < prefix_fraction <= 1:
        raise ValueError("prefix_fraction must be in (0, 1]")
    if not np.isfinite(adf_p) or not 0 < adf_p < 1:
        raise ValueError("adf_p must be between 0 and 1")
    if candidate_values.ndim != 1 or not len(candidate_values):
        raise ValueError("candidates must contain at least one d")
    if not np.isfinite(candidate_values).all() or np.any(candidate_values < 0):
        raise ValueError("candidate d values must be finite and non-negative")

    prefixes = []
    for start, end in sequence_boundaries:
        if start < 0 or end > len(values) or start >= end:
            raise ValueError("invalid sequence boundary")
        if start >= train_events:
            continue
        if end > train_events:
            raise ValueError("a stock sequence crosses the train/test boundary")
        prefix_end = start + int((end - start) * prefix_fraction)
        prefix = values[start:prefix_end]
        if len(prefix) >= max(100, width + 100):
            prefixes.append({"start": int(start), "end": int(prefix_end), "values": prefix})
    if not prefixes:
        raise ValueError("no sufficiently long training-only stock prefixes")

    rows = []
    for d in candidate_values:
        pvalues = []
        correlations = []
        for segment in prefixes:
            original = segment["values"]
            transformed = fractional_diff_cpu(original, float(d), width)
            finite = transformed[np.isfinite(transformed)]
            if len(finite) < 100:
                continue
            pvalue = float(adfuller(
                finite, maxlag=min(20, len(finite) // 10), autolag="AIC"
            )[1])
            corr = (
                float(np.corrcoef(original[-len(finite):], finite)[0, 1])
                if np.std(finite) and np.std(original[-len(finite):]) else 0.0
            )
            pvalues.append(pvalue)
            correlations.append(corr)
        if not pvalues:
            continue
        rows.append({
            "d": float(d),
            "adf_pvalue": float(np.median(pvalues)),
            "median_adf_pvalue": float(np.median(pvalues)),
            "segments_tested": int(len(pvalues)),
            "segments_p_below_threshold": int(np.sum(np.asarray(pvalues) < adf_p)),
            "corr_with_price": float(np.median(correlations)),
            "segment_adf_pvalues": pvalues,
        })
    if not rows:
        raise ValueError("could not evaluate any FFD order on the training prefixes")
    chosen = next((r["d"] for r in rows if r["median_adf_pvalue"] < adf_p), rows[-1]["d"])
    return float(chosen), {
        "candidates": rows,
        "selected_d": float(chosen),
        "adf_threshold": float(adf_p),
        "selection_mode": "median ADF p-value across separate training-only stock prefixes",
        "prefix_fraction_per_stock": float(prefix_fraction),
        "training_segments_tested": int(len(prefixes)),
        "training_prefixes": [
            {"start": x["start"], "end": x["end"], "events": int(len(x["values"]))}
            for x in prefixes
        ],
    }

def microstructure_by_segment(
    frames: Sequence[pd.DataFrame], depth_levels: int = 5
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Build rolling returns, OFI and volatility separately for each stock sequence."""
    boundaries = segment_boundaries(frames)
    parts = [microstructure_features(frame, depth_levels=depth_levels) for frame in frames]
    return pd.concat(parts, ignore_index=True), boundaries


def fractional_diff_by_segment(
    log_mid: np.ndarray,
    boundaries: Sequence[tuple[int, int]],
    d: float,
    width: int,
    *,
    require_cuda: bool = False,
) -> tuple[np.ndarray, bool, str]:
    """Reset the causal FFD filter at each stock-sequence boundary."""
    values = np.asarray(log_mid, dtype=float)
    outputs, backends, used_flags = [], [], []
    for start, end in boundaries:
        if start < 0 or end > len(values) or start >= end:
            raise ValueError("invalid segment boundary")
        out, used_gpu, backend = fractional_diff_gpu(values[start:end], d, width)
        outputs.append(out)
        used_flags.append(bool(used_gpu))
        backends.append(backend)
    used_gpu = bool(used_flags) and all(used_flags)
    if require_cuda and not used_gpu:
        raise RuntimeError("CUDA was required but at least one segment used CPU fallback")
    if used_gpu and all(x == "numba-cuda" for x in backends):
        backend_name = "numba-cuda"
    elif len(set(backends)) == 1:
        backend_name = backends[0]
    else:
        backend_name = "mixed:" + ",".join(sorted(set(backends)))
    return np.concatenate(outputs), used_gpu, backend_name


def topology_by_segment(
    frames: Sequence[pd.DataFrame],
    boundaries: Sequence[tuple[int, int]],
    radii: np.ndarray,
    *,
    horizon: int,
    stride: int,
    levels: int = 10,
    require_gtda: bool = True,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Create clouds from each stock sequence and fill values only within it."""
    if horizon < 1 or stride < 1 or levels < 1:
        raise ValueError("horizon, stride and levels must be positive")
    if len(frames) != len(boundaries):
        raise ValueError("one boundary is required for each frame")

    global_indices, clouds = [], []
    for frame, (start, end) in zip(frames, boundaries):
        if end - start != len(frame):
            raise ValueError("frame length does not match its global boundary")
        local_indices = np.arange(50, len(frame) - horizon, stride, dtype=int)
        if len(local_indices):
            clouds.extend(make_point_clouds(frame, local_indices, levels=levels))
            global_indices.extend((start + local_indices).tolist())

    indices = np.asarray(global_indices, dtype=int)
    if not clouds:
        raise ValueError("segments are too short to create topology clouds")
    features, meta = persistent_features(
        clouds,
        np.asarray(radii, dtype=float),
        require_gtda=require_gtda,
        n_jobs=n_jobs,
    )
    if len(features) != len(indices):
        raise RuntimeError("TDA feature count does not match cloud indices")

    result = pd.DataFrame(
        features, index=indices, columns=meta["feature_names"]
    ).reindex(range(sum(len(frame) for frame in frames)))
    for start, end in boundaries:
        result.iloc[start:end] = result.iloc[start:end].ffill()

    meta = dict(meta)
    meta["n_clouds"] = int(len(clouds))
    meta["segment_cloud_counts"] = [
        int(np.sum((indices >= start) & (indices < end)))
        for start, end in boundaries
    ]
    return result, meta, indices
