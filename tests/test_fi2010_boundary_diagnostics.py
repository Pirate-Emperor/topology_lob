import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_fi2010_boundaries import discontinuity_cuts, forward_returns, return_summary


def canonical_frames(mids):
    rows = []
    for i, mid in enumerate(mids):
        row = {"timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(microseconds=i)}
        for level in range(1, 3):
            tick = 0.01 * level
            row[f"bid_price_{level}"] = mid - tick
            row[f"ask_price_{level}"] = mid + tick
            row[f"bid_size_{level}"] = 20.0
            row[f"ask_size_{level}"] = 21.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_price_discontinuity_cut_finds_large_jump_without_short_fragments():
    mids = np.r_[np.full(1500, 10.0), np.full(1500, 30.0)]
    cuts, details = discontinuity_cuts(canonical_frames(mids), threshold_pct=0.02, min_segment_events=500)
    assert cuts == [1500]
    assert details[0]["absolute_return_pct"] > 100


def test_price_discontinuity_cut_ignores_short_or_small_moves():
    mids = np.r_[np.full(900, 10.0), np.full(100, 10.1), np.full(900, 10.2)]
    cuts, _ = discontinuity_cuts(canonical_frames(mids), threshold_pct=0.02, min_segment_events=500)
    assert cuts == []
    with pytest.raises(ValueError):
        discontinuity_cuts(canonical_frames(mids), threshold_pct=0.0, min_segment_events=500)


def test_forward_return_summaries_drop_cross_cut_targets():
    values = np.log(np.r_[np.full(50, 10.0), np.full(50, 30.0)])
    full = forward_returns(values, [(0, len(values))], 10)
    cut = forward_returns(values, [(0, 50), (50, 100)], 10)
    assert np.nanmax(np.abs(full)) > 1.0
    assert np.isnan(cut[40:50]).all()
    assert return_summary(cut)["count_abs_ge_10pct"] == 0
