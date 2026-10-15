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


def microstructure_by_segment(
    frames: Sequence[pd.DataFrame], depth_levels: int = 5
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Build rolling returns, OFI and volatility separately for each file."""
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
    """Reset the causal FFD filter at each file boundary."""
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
    """Create clouds from each file and fill topology values within that file only."""
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
