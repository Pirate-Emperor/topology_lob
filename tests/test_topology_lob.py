import numpy as np
import xgboost as xgb

from topology_lob import (
    LOBValidationError,
    fracdiff_weights,
    fractional_diff_cpu,
    fractional_diff_gpu,
    microstructure_features,
    select_stationary_d,
    focal_grad_hess,
    focal_loss_value,
    fit_focal_xgb,
    make_point_clouds,
    make_synthetic_lob,
    persistent_features,
    validate_lob,
)


def test_schema_and_timestamps():
    df = make_synthetic_lob(200, 10, 1)
    assert df["timestamp"].is_monotonic_increasing
    assert "bid_price_10" in df


def test_malformed_books_rejected():
    df = make_synthetic_lob(20, 4, 2)

    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(ask_price_1=df.bid_price_1))

    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(bid_size_1=-1.0))

    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.iloc[[0, 0]].copy())


def test_ffd_d0_identity_and_finite_weights():
    x = np.arange(100, dtype=float)
    np.testing.assert_allclose(fractional_diff_cpu(x, 0, 32)[31:], x[31:])
    weights = fracdiff_weights(0.45, 64)
    assert len(weights) == 64
    assert np.isfinite(weights).all()


def test_ffd_is_causal():
    x = np.linspace(1.0, 2.0, 400)
    changed = x.copy()
    changed[300:] += 10.0
    left = fractional_diff_cpu(x, 0.45, 64)
    right = fractional_diff_cpu(changed, 0.45, 64)
    np.testing.assert_allclose(left[:300], right[:300], atol=0.0, rtol=0.0)


def test_topology_detects_gap():
    left = np.column_stack([np.linspace(-2, -0.5, 8), np.zeros(8)])
    right = np.column_stack([np.linspace(0.5, 2, 8), np.zeros(8)])
    features, meta = persistent_features(
        [np.vstack([left, right])],
        np.array([0.1, 0.5, 1.0]),
        require_gtda=False,
    )
    assert meta["backend"] in {"exact-vr-gf2-fallback", "giotto-tda"}
    assert features[0, 0] >= 2


def test_vr_square_cycle_is_killed_by_clique_filling():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    feats, meta = persistent_features(
        [square],
        np.array([1.01, 1.5]),
        require_gtda=False,
    )
    assert meta["backend"] in {"exact-vr-gf2-fallback", "giotto-tda"}
    if meta["backend"] == "exact-vr-gf2-fallback":
        assert feats[0, 2] == 1
        assert feats[0, 3] == 0
    else:
        assert np.isfinite(feats).all()


def test_topology_is_invariant_to_common_price_translation():
    df = make_synthetic_lob(30, 4, 3)
    shifted = df.copy()
    for level in range(1, 5):
        shifted[f"bid_price_{level}"] += 25.0
        shifted[f"ask_price_{level}"] += 25.0

    c1 = make_point_clouds(df, np.array([10]), 4)[0]
    c2 = make_point_clouds(shifted, np.array([10]), 4)[0]
    np.testing.assert_allclose(c1, c2, atol=0.0, rtol=0.0)


def test_focal_gradient_and_hessian_against_finite_difference():
    pred = np.array([-1.1, 0.2, 1.7])
    y = np.array([0.0, 1.0, 1.0])
    dtrain = xgb.DMatrix(np.zeros((3, 1)), label=y)
    grad, hess = focal_grad_hess(pred.copy(), dtrain)

    eps = 1e-5
    max_grad = 0.0
    max_hess = 0.0
    for i in range(3):
        plus = pred.copy()
        minus = pred.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric_grad = (
            focal_loss_value(plus, y) - focal_loss_value(minus, y)
        ) / (2 * eps) * len(y)
        max_grad = max(max_grad, abs(grad[i] - numeric_grad))

        gplus, _ = focal_grad_hess(plus, dtrain)
        gminus, _ = focal_grad_hess(minus, dtrain)
        numeric_hess = (gplus[i] - gminus[i]) / (2 * eps)
        max_hess = max(max_hess, abs(hess[i] - numeric_hess))

    assert max_grad < 1e-7
    assert max_hess < 1e-6


def test_focal_prediction_is_explicitly_logistic():
    df = make_synthetic_lob(500, 10, 3)
    feature = df["bid_size_1"].to_numpy().reshape(-1, 1)
    y = (np.arange(len(df)) % 5 == 0).astype(int)
    model = fit_focal_xgb(feature[:400], y[:400], seed=11)
    margin = model.predict(xgb.DMatrix(feature[400:]), output_margin=True)
    probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
    assert np.isfinite(margin).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert not np.array_equal(np.round(margin, 6), np.round(probability, 6))


def test_missing_timestamp_empty_input_and_nonfinite_values_rejected():
    df = make_synthetic_lob(30, 4, 31)
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.drop(columns=["timestamp"]))
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.iloc[:0])
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(bid_size_1=np.inf))
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(ask_price_1=-1.0))


def test_price_levels_must_be_ordered():
    df = make_synthetic_lob(30, 4, 32)
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(bid_price_2=df.bid_price_1 + 0.01))
    with np.testing.assert_raises(LOBValidationError):
        validate_lob(df.assign(ask_price_2=df.ask_price_1 - 0.01))


def test_feature_depth_and_fractional_difference_parameters_are_checked():
    df = make_synthetic_lob(40, 4, 33)
    with np.testing.assert_raises(ValueError):
        microstructure_features(df, 0)
    with np.testing.assert_raises(ValueError):
        fracdiff_weights(float("nan"), 32)
    with np.testing.assert_raises(ValueError):
        fracdiff_weights(0.45, 0)
    with np.testing.assert_raises(ValueError):
        fracdiff_weights(-0.1, 32)


def test_stationarity_selection_rejects_empty_or_invalid_candidates():
    x = np.linspace(1.0, 2.0, 400)
    with np.testing.assert_raises(ValueError):
        select_stationary_d(x, [])
    with np.testing.assert_raises(ValueError):
        select_stationary_d(x, [0.1, float("nan")])
    with np.testing.assert_raises(ValueError):
        select_stationary_d(x, [0.1], adf_p=1.5)


def test_fractional_difference_gpu_matches_cpu_when_available():
    x = np.linspace(-0.5, 0.8, 600)
    expected = fractional_diff_cpu(x, 0.45, 64)
    actual, used_gpu, backend = fractional_diff_gpu(x, 0.45, 64)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    if used_gpu:
        assert backend == "numba-cuda"
    else:
        assert backend.startswith("cpu-fallback:")
