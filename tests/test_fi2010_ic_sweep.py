import numpy as np

from scripts.run_fi2010_ic_sweep import binary_labels, corr, forward_returns


def test_segment_forward_returns_never_crosses_boundaries():
    log_mid = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    result = forward_returns(log_mid, [(0, 3), (3, 6)], horizon=1)
    np.testing.assert_allclose(result[[0, 1, 3, 4]], [0.1, 0.1, 0.1, 0.1])
    assert np.isnan(result[2])
    assert np.isnan(result[5])
    assert result[1] != log_mid[3] - log_mid[1]


def test_binary_labels_keep_nonfinite_rows_unlabelled():
    future = np.asarray([0.2, 0.05, -0.1, np.nan])
    result = binary_labels(future, threshold=0.05)
    np.testing.assert_array_equal(result, [1, 0, 0, -1])


def test_corr_handles_constant_and_mismatched_inputs():
    assert corr(np.ones(4), np.arange(4.0)) == 0.0
    assert corr(np.arange(4.0), np.arange(3.0)) == 0.0
    assert np.isclose(corr(np.arange(4.0), np.arange(4.0)), 1.0)


from scripts.run_fi2010_return_regression import (
    correlation as regression_correlation,
    forward_returns as regression_forward_returns,
)


def test_regression_forward_returns_never_cross_daily_boundaries():
    log_mid = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 20.0, 20.1, 20.2])
    boundaries = [(0, 3), (3, 6), (6, 9)]
    result = regression_forward_returns(log_mid, boundaries, horizon=2)
    np.testing.assert_allclose(result[[0, 3, 6]], [0.2, 0.2, 0.2], atol=1e-12)
    assert np.isnan(result[[1, 2, 4, 5, 7, 8]]).all()


def test_regression_correlation_checks_finite_values():
    assert np.isclose(regression_correlation(np.arange(5.0), np.arange(5.0)), 1.0)
    assert regression_correlation(np.ones(5), np.arange(5.0)) == 0.0
    with np.testing.assert_raises(ValueError):
        regression_correlation(np.asarray([1.0, np.nan]), np.asarray([0.0, 1.0]))


from scripts.run_fi2010_focal_horizon_selection import direction_labels, validate_candidate


def test_focal_horizon_labels_mark_missing_returns_invalid():
    values = np.asarray([0.001, 0.0, -0.001, np.nan])
    np.testing.assert_array_equal(direction_labels(values), [1, 0, 0, -1])


def test_focal_horizon_candidate_validation_rejects_bad_config():
    validate_candidate({
        "horizon": 1000,
        "feature_set": "micro_only",
        "validation": {"pearson_ic": 0.01, "rank_ic": 0.02},
    })
    with np.testing.assert_raises(ValueError):
        validate_candidate({"horizon": 11, "feature_set": "micro_only", "validation": {}})
    with np.testing.assert_raises(ValueError):
        validate_candidate({"horizon": 10, "feature_set": "unknown", "validation": {}})


from scripts.run_fi2010_focal_horizon_selection import split_positions


def test_focal_horizon_split_filters_nonfinite_feature_rows():
    features = np.asarray([[1.0], [np.nan], [3.0], [4.0], [np.inf], [6.0], [7.0], [8.0]])
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    indices = np.arange(8)
    fit, validation = split_positions(
        features, labels, indices,
        fit_end=5, validation_start=5, train_events=8, horizon=1, purge=1,
    )
    np.testing.assert_array_equal(fit, [0, 2, 3])
    np.testing.assert_array_equal(validation, [6])
    assert np.isfinite(features[fit]).all()
    assert np.isfinite(features[validation]).all()
