import numpy as np
import pandas as pd

from scripts.fi2010_feature_helpers import (
    select_stationary_d_by_segment,
    split_stock_sequences,
)


def canonical_frame(mids):
    mids = np.asarray(mids, dtype=float)
    rows = []
    for i, mid in enumerate(mids):
        row = {"timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(microseconds=i)}
        for level in range(1, 3):
            offset = 0.01 * level
            row[f"bid_price_{level}"] = mid - offset
            row[f"ask_price_{level}"] = mid + offset
            row[f"bid_size_{level}"] = 10.0 + level
            row[f"ask_size_{level}"] = 12.0 + level
        rows.append(row)
    return pd.DataFrame(rows)


def test_fi2010_splits_five_concatenated_stock_streams_per_file():
    # Each block has a stable price scale; large jumps occur only between stocks.
    blocks = [
        np.linspace(10.0, 10.2, 40),
        np.linspace(22.0, 22.2, 40),
        np.linspace(7.0, 7.1, 40),
        np.linspace(35.0, 35.4, 40),
        np.linspace(14.0, 14.3, 40),
    ]
    train = canonical_frame(np.concatenate(blocks))
    test = canonical_frame(np.concatenate([np.asarray(x) * 1.01 for x in blocks]))
    sequences, file_bounds, seq_bounds, cuts = split_stock_sequences(
        [train, test], threshold_pct=0.05, min_segment_events=20
    )

    assert file_bounds == [(0, 200), (200, 400)]
    assert len(sequences) == 10
    assert len(seq_bounds) == 10
    assert [b[1] - b[0] for b in seq_bounds] == [40] * 10
    assert [c["file_index"] for c in cuts] == [0] * 4 + [1] * 4
    assert [c["file_row"] for c in cuts] == [40, 80, 120, 160] * 2
    assert sum(len(x) for x in sequences) == len(train) + len(test)


def test_jump_split_ignores_small_price_changes_and_never_crosses_file_edges():
    frame_a = canonical_frame(np.linspace(10.0, 10.3, 300))
    frame_b = canonical_frame(np.linspace(10.2, 10.5, 300))
    sequences, file_bounds, seq_bounds, cuts = split_stock_sequences(
        [frame_a, frame_b], threshold_pct=0.05, min_segment_events=100
    )
    assert len(sequences) == 2
    assert file_bounds == [(0, 300), (300, 600)]
    assert seq_bounds == file_bounds
    assert cuts == []


def test_stationarity_selection_uses_training_stock_prefixes_only():
    rng = np.random.default_rng(29)
    train_series = [
        4.0 + np.cumsum(rng.normal(0.0, 0.01, 700)),
        8.0 + np.cumsum(rng.normal(0.0, 0.012, 700)),
        3.0 + np.cumsum(rng.normal(0.0, 0.008, 700)),
    ]
    test_series = [1000.0 + np.cumsum(rng.normal(0.0, 0.02, 700))]
    series = np.concatenate([*train_series, *test_series])
    boundaries = [(0, 700), (700, 1400), (1400, 2100), (2100, 2800)]
    d1, report1 = select_stationary_d_by_segment(
        series, boundaries, 2100, [0.0, 0.1, 0.2, 0.3],
        adf_p=0.05, width=32, prefix_fraction=0.6,
    )
    changed_test = series.copy()
    changed_test[2100:] *= 10000.0
    d2, report2 = select_stationary_d_by_segment(
        changed_test, boundaries, 2100, [0.0, 0.1, 0.2, 0.3],
        adf_p=0.05, width=32, prefix_fraction=0.6,
    )

    assert d1 == d2
    assert report1["training_segments_tested"] == 3
    assert report2["training_segments_tested"] == 3
    assert report1["candidates"] == report2["candidates"]
    assert all(x["end"] <= 2100 for x in report1["training_prefixes"])
