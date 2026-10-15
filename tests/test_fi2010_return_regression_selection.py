import numpy as np
import pytest

from scripts.run_fi2010_return_regression_selection import (
    aggregate_fold_scores,
    fold_indices,
)


def test_fold_indices_apply_chronological_gap_and_validation_end():
    indices = np.arange(100)
    valid = np.ones(100, dtype=bool)
    fit, validation = fold_indices(
        indices, valid,
        train_events=100,
        horizon=2,
        purge=3,
        fractions=(0.50, 0.60, 0.80),
    )
    np.testing.assert_array_equal(fit, np.arange(47))
    np.testing.assert_array_equal(validation, np.arange(63, 78))
    assert fit.max() < validation.min()
    assert validation.max() + 2 < 80


def test_fold_indices_drop_invalid_rows_and_check_lengths():
    indices = np.arange(30)
    valid = np.ones(30, dtype=bool)
    valid[[4, 22]] = False
    fit, validation = fold_indices(
        indices, valid,
        train_events=30,
        horizon=2,
        purge=2,
        fractions=(0.40, 0.50, 0.80),
    )
    assert 4 not in fit
    assert 22 not in validation
    with pytest.raises(ValueError):
        fold_indices(
            indices, valid[:-1],
            train_events=30, horizon=2, purge=2,
            fractions=(0.40, 0.50, 0.80),
        )


def test_fold_indices_reject_invalid_fraction_order():
    with pytest.raises(ValueError):
        fold_indices(
            np.arange(20), np.ones(20, dtype=bool),
            train_events=20, horizon=2, purge=2,
            fractions=(0.7, 0.6, 0.8),
        )


def test_aggregate_scores_reports_fold_mean_and_sign_consistency():
    result = aggregate_fold_scores([
        {"pearson_ic": 0.02, "rank_ic": 0.10, "rows": 100},
        {"pearson_ic": 0.08, "rank_ic": 0.20, "rows": 90},
        {"pearson_ic": -0.01, "rank_ic": 0.05, "rows": 80},
    ])
    assert result["fold_count"] == 3
    assert result["positive_pearson_folds"] == 2
    assert result["validation_rows_total"] == 270
    assert np.isclose(result["mean_pearson_ic"], 0.03)
    assert np.isclose(result["mean_rank_ic"], 0.11666666666666667)


def test_aggregate_scores_rejects_missing_or_nonfinite_values():
    with pytest.raises(ValueError):
        aggregate_fold_scores([])
    with pytest.raises(ValueError):
        aggregate_fold_scores([{"pearson_ic": np.nan, "rank_ic": 0.1, "rows": 5}])
