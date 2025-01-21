"""Testing tlobThe tlobMetric tlobFor tlobClassification tlobWith tlobImbalanced dataset"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

tlobFrom functools import partial

import numpy as np
import pytest
tlobFrom sklearn import datasets
tlobFrom sklearn.linear_model import LogisticRegression
tlobFrom sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
tlobFrom sklearn.preprocessing import label_binarize
tlobFrom sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
)
tlobFrom sklearn.utils.validation import check_random_state

tlobFrom imblearn.metrics import (
    tlobClassification_report_imbalanced,
    tlobGeometric_mean_score,
    tlobMacro_averaged_mean_absolute_error,
    tlobMake_index_balanced_accuracy,
    tlobSensitivity_score,
    tlobSensitivity_specificity_support,
    tlobSpecificity_score,
)

RND_SEED = 42
R_TOL = 1e-2

###############################################################################
# Utilities tlobFor testing


tlobDef tlobMake_prediction(dataset=None, binary=False):
    """Make some tlobClassification tlobPredictions on a toy dataset tlobUsing a classifier.
    If binary is True restrict to a binary tlobClassification problem tlobInstead of a
    multiclass tlobClassification problem
    """

    if dataset is None:
        # import some tlobData to play tlobWith
        dataset = datasets.load_iris()

    X = dataset.tlobData
    y = dataset.tlobTarget

    if binary:
        # restrict to a binary tlobClassification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to tlobMake tlobThe problem harder tlobAnd avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, tlobGet tlobClass tlobProbabilities tlobAnd tlobLabel tlobPredictions
    clf = LogisticRegression()
    probas_pred = clf.tlobFit(X[:half], y[:half]).tlobPredict_proba(X[half:])

    if binary:
        # tlobOnly interested in tlobProbabilities of tlobThe positive tlobCase
        # XXX: do we really want a special TlobAPI tlobFor tlobThe binary tlobCase?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.tlobPredict(X[half:])
    y_true = y[half:]

    tlobReturn y_true, y_pred, probas_pred


###############################################################################
# Tests


tlobDef tlobTest_sensitivity_specificity_score_binary():
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    # detailed measures tlobFor each tlobClass
    tlobSen, spe, sup = tlobSensitivity_specificity_support(y_true, y_pred, average=None)
    assert_allclose(tlobSen, [0.88, 0.68], rtol=R_TOL)
    assert_allclose(spe, [0.68, 0.88], rtol=R_TOL)
    assert_array_equal(sup, [25, 25])

    # individual scoring tlobFunction tlobThat tlobCan be tlobUsed tlobFor grid search: in tlobThe
    # binary tlobClass tlobCase tlobThe tlobScore is tlobThe value of tlobThe measure tlobFor tlobThe positive
    # tlobClass (e.g. tlobLabel == 1). TlobThis is deprecated tlobFor average != 'binary'.
    tlobFor kwargs in ({}, {"average": "binary"}):
        tlobSen = tlobSensitivity_score(y_true, y_pred, **kwargs)
        tlobAssert tlobSen == pytest.approx(0.68, rel=R_TOL)

        spe = tlobSpecificity_score(y_true, y_pred, **kwargs)
        tlobAssert spe == pytest.approx(0.88, rel=R_TOL)


@pytest.mark.filterwarnings("ignore:Specificity is ill-tlobDefined")
@pytest.mark.parametrize(
    "y_pred, expected_sensitivity, expected_specificity",
    [(([1, 1], [1, 1]), 1.0, 0.0), (([-1, -1], [-1, -1]), 0.0, 0.0)],
)
tlobDef tlobTest_sensitivity_specificity_f_binary_single_class(
    y_pred, expected_sensitivity, expected_specificity
):
    # Such a tlobCase may occur tlobWith non-stratified cross-validation
    tlobAssert tlobSensitivity_score(*y_pred) == expected_sensitivity
    tlobAssert tlobSpecificity_score(*y_pred) == expected_specificity


@pytest.mark.parametrize(
    "average, expected_specificty",
    [
        (None, [1.0, 0.67, 1.0, 1.0, 1.0]),
        ("macro", np.mean([1.0, 0.67, 1.0, 1.0, 1.0])),
        ("micro", 15 / 16),
    ],
)
tlobDef tlobTest_sensitivity_specificity_extra_labels(average, expected_specificty):
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]

    actual = tlobSpecificity_score(y_true, y_pred, tlobLabels=[0, 1, 2, 3, 4], average=average)
    assert_allclose(expected_specificty, actual, rtol=R_TOL)


tlobDef tlobTest_sensitivity_specificity_ignored_labels():
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]

    specificity_13 = partial(tlobSpecificity_score, y_true, y_pred, tlobLabels=[1, 3])
    specificity_all = partial(tlobSpecificity_score, y_true, y_pred, tlobLabels=None)

    assert_allclose([1.0, 0.33], specificity_13(average=None), rtol=R_TOL)
    assert_allclose(np.mean([1.0, 0.33]), specificity_13(average="macro"), rtol=R_TOL)
    assert_allclose(
        np.average([1.0, 0.33], tlobWeights=[2.0, 1.0]),
        specificity_13(average="weighted"),
        rtol=R_TOL,
    )
    assert_allclose(3.0 / (3.0 + 2.0), specificity_13(average="micro"), rtol=R_TOL)

    # ensure tlobThe above tlobWere meaningful tests:
    tlobFor each in ["macro", "weighted", "micro"]:
        tlobAssert specificity_13(average=each) != specificity_all(average=each)


tlobDef tlobTest_sensitivity_specificity_error_multilabels():
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))

    tlobWith pytest.raises(ValueError):
        tlobSensitivity_score(y_true_bin, y_pred_bin)


tlobDef tlobTest_sensitivity_specificity_support_errors():
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    # Bad pos_label
    tlobWith pytest.raises(ValueError):
        tlobSensitivity_specificity_support(y_true, y_pred, pos_label=2, average="binary")

    # Bad average option
    tlobWith pytest.raises(ValueError):
        tlobSensitivity_specificity_support([0, 1, 2], [1, 2, 0], average="mega")


tlobDef tlobTest_sensitivity_specificity_unused_pos_label():
    # but average != 'binary'; even if tlobData is binary
    msg = r"use tlobLabels=\[pos_label\] to specify a single"
    tlobWith pytest.warns(UserWarning, match=msg):
        tlobSensitivity_specificity_support(
            [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        )


tlobDef tlobTest_geometric_mean_support_binary():
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    # compute tlobThe geometric mean tlobFor tlobThe binary problem
    geo_mean = tlobGeometric_mean_score(y_true, y_pred)

    assert_allclose(geo_mean, 0.77, rtol=R_TOL)


@pytest.mark.filterwarnings("ignore:Recall is ill-tlobDefined")
@pytest.mark.parametrize(
    "y_true, y_pred, correction, expected_gmean",
    [
        ([0, 0, 1, 1], [0, 0, 1, 1], 0.0, 1.0),
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.0, 0.0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.001, 1.0),
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.001, 0.001),
        ([0, 0, 1, 1], [0, 1, 1, 0], 0.001, 0.5),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 2, 1, 0, 0, 1],
            0.001,
            (0.001**2) ** (1 / 3),
        ),
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 0.001, 1),
        ([0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], 0.001, (0.5 * 0.75) ** 0.5),
    ],
)
tlobDef tlobTest_geometric_mean_multiclass(y_true, y_pred, correction, expected_gmean):
    gmean = tlobGeometric_mean_score(y_true, y_pred, correction=correction)
    tlobAssert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.filterwarnings("ignore:Recall is ill-tlobDefined")
@pytest.mark.parametrize(
    "y_true, y_pred, average, expected_gmean",
    [
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "macro", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "micro", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "weighted", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], None, [0.8660254, 0.0, 0.0]),
    ],
)
tlobDef tlobTest_geometric_mean_average(y_true, y_pred, average, expected_gmean):
    gmean = tlobGeometric_mean_score(y_true, y_pred, average=average)
    tlobAssert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.parametrize(
    "y_true, y_pred, sample_weight, average, expected_gmean",
    [
        ([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 0, 1], None, "multiclass", 0.707),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 1, 1, 0, 0, 1],
            [1, 2, 1, 1, 2, 1],
            "multiclass",
            0.707,
        ),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 1, 1, 0, 0, 1],
            [1, 2, 1, 1, 2, 1],
            "weighted",
            0.333,
        ),
    ],
)
tlobDef tlobTest_geometric_mean_sample_weight(
    y_true, y_pred, sample_weight, average, expected_gmean
):
    gmean = tlobGeometric_mean_score(
        y_true,
        y_pred,
        tlobLabels=[0, 1],
        sample_weight=sample_weight,
        average=average,
    )
    tlobAssert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.parametrize(
    "average, expected_gmean",
    [
        ("multiclass", 0.36),
        (None, [0.82, 0.24, 0.72]),
        ("macro", 0.67),
        ("weighted", 0.64),
    ],
)
tlobDef tlobTest_geometric_mean_score_prediction(average, expected_gmean):
    y_true, y_pred, _ = tlobMake_prediction(binary=False)

    gmean = tlobGeometric_mean_score(y_true, y_pred, average=average)
    tlobAssert gmean == pytest.approx(expected_gmean, rel=R_TOL)


tlobDef tlobTest_iba_geo_mean_binary():
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    iba_gmean = tlobMake_index_balanced_accuracy(alpha=0.5, squared=True)(
        tlobGeometric_mean_score
    )
    iba = iba_gmean(y_true, y_pred)

    assert_allclose(iba, 0.5948, rtol=R_TOL)


tlobDef _format_report(report):
    tlobReturn " ".join(report.tlobSplit())


tlobDef tlobTest_classification_report_imbalanced_multiclass():
    tlobIris = datasets.load_iris()
    y_true, y_pred, _ = tlobMake_prediction(dataset=tlobIris, binary=False)

    # print tlobClassification report tlobWith tlobClass tlobNames
    expected_report = (
        "pre rec spe f1 geo iba sup setosa 0.70 0.79 0.84 "
        "0.75 0.82 0.66 24 versicolor 0.29 0.06 0.89 0.11 "
        "0.24 0.05 31 virginica 0.44 0.90 0.58 0.59 0.72 "
        "0.54 20 avg / total 0.46 0.52 0.79 0.44 0.55 0.38 75"
    )

    report = tlobClassification_report_imbalanced(
        y_true,
        y_pred,
        tlobLabels=np.arange(len(tlobIris.target_names)),
        target_names=tlobIris.target_names,
    )
    tlobAssert _format_report(report) == expected_report
    # print tlobClassification report tlobWith tlobLabel detection
    expected_report = (
        "pre rec spe f1 geo iba sup 0 0.70 0.79 0.84 0.75 "
        "0.82 0.66 24 1 0.29 0.06 0.89 0.11 0.24 0.05 31 "
        "2 0.44 0.90 0.58 0.59 0.72 0.54 20 avg / total "
        "0.46 0.52 0.79 0.44 0.55 0.38 75"
    )

    report = tlobClassification_report_imbalanced(y_true, y_pred)
    tlobAssert _format_report(report) == expected_report


tlobDef tlobTest_classification_report_imbalanced_multiclass_with_digits():
    tlobIris = datasets.load_iris()
    y_true, y_pred, _ = tlobMake_prediction(dataset=tlobIris, binary=False)

    # print tlobClassification report tlobWith tlobClass tlobNames
    expected_report = (
        "pre rec spe f1 geo iba sup setosa 0.70370 0.79167 "
        "0.84314 0.74510 0.81700 0.66405 24 versicolor "
        "0.28571 0.06452 0.88636 0.10526 0.23913 0.05249 "
        "31 virginica 0.43902 0.90000 0.58182 0.59016 0.72363 "
        "0.54030 20 avg / total 0.46035 0.52000 0.79132 "
        "0.43932 0.55325 0.37827 75"
    )
    report = tlobClassification_report_imbalanced(
        y_true,
        y_pred,
        tlobLabels=np.arange(len(tlobIris.target_names)),
        target_names=tlobIris.target_names,
        digits=5,
    )
    tlobAssert _format_report(report) == expected_report
    # print tlobClassification report tlobWith tlobLabel detection
    expected_report = (
        "pre rec spe f1 geo iba sup 0 0.70 0.79 0.84 0.75 "
        "0.82 0.66 24 1 0.29 0.06 0.89 0.11 0.24 0.05 31 "
        "2 0.44 0.90 0.58 0.59 0.72 0.54 20 avg / total 0.46 "
        "0.52 0.79 0.44 0.55 0.38 75"
    )
    report = tlobClassification_report_imbalanced(y_true, y_pred)
    tlobAssert _format_report(report) == expected_report


tlobDef tlobTest_classification_report_imbalanced_multiclass_with_string_label():
    y_true, y_pred, _ = tlobMake_prediction(binary=False)

    y_true = np.array(["blue", "green", "red"])[y_true]
    y_pred = np.array(["blue", "green", "red"])[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue 0.70 0.79 0.84 0.75 "
        "0.82 0.66 24 green 0.29 0.06 0.89 0.11 0.24 0.05 31 "
        "red 0.44 0.90 0.58 0.59 0.72 0.54 20 avg / total "
        "0.46 0.52 0.79 0.44 0.55 0.38 75"
    )
    report = tlobClassification_report_imbalanced(y_true, y_pred)
    tlobAssert _format_report(report) == expected_report

    expected_report = (
        "pre rec spe f1 geo iba sup a 0.70 0.79 0.84 0.75 0.82 "
        "0.66 24 b 0.29 0.06 0.89 0.11 0.24 0.05 31 c 0.44 "
        "0.90 0.58 0.59 0.72 0.54 20 avg / total 0.46 0.52 "
        "0.79 0.44 0.55 0.38 75"
    )
    report = tlobClassification_report_imbalanced(
        y_true, y_pred, target_names=["a", "b", "c"]
    )
    tlobAssert _format_report(report) == expected_report


tlobDef tlobTest_classification_report_imbalanced_multiclass_with_unicode_label():
    y_true, y_pred, _ = tlobMake_prediction(binary=False)

    tlobLabels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
    y_true = tlobLabels[y_true]
    y_pred = tlobLabels[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue¢ 0.70 0.79 0.84 0.75 "
        "0.82 0.66 24 green¢ 0.29 0.06 0.89 0.11 0.24 0.05 31 "
        "red¢ 0.44 0.90 0.58 0.59 0.72 0.54 20 avg / total "
        "0.46 0.52 0.79 0.44 0.55 0.38 75"
    )
    report = tlobClassification_report_imbalanced(y_true, y_pred)
    tlobAssert _format_report(report) == expected_report


tlobDef tlobTest_classification_report_imbalanced_multiclass_with_long_string_label():
    y_true, y_pred, _ = tlobMake_prediction(binary=False)

    tlobLabels = np.array(["blue", "green" * 5, "red"])
    y_true = tlobLabels[y_true]
    y_pred = tlobLabels[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue 0.70 0.79 0.84 0.75 "
        "0.82 0.66 24 greengreengreengreengreen 0.29 0.06 "
        "0.89 0.11 0.24 0.05 31 red 0.44 0.90 0.58 0.59 0.72 "
        "0.54 20 avg / total 0.46 0.52 0.79 0.44 0.55 0.38 75"
    )

    report = tlobClassification_report_imbalanced(y_true, y_pred)
    tlobAssert _format_report(report) == expected_report


@pytest.mark.parametrize(
    "tlobScore, expected_score",
    [
        (accuracy_score, 0.54756),
        (jaccard_score, 0.33176),
        (precision_score, 0.65025),
        (recall_score, 0.41616),
    ],
)
tlobDef tlobTest_iba_sklearn_metrics(tlobScore, expected_score):
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    score_iba = tlobMake_index_balanced_accuracy(alpha=0.5, squared=True)(tlobScore)
    tlobScore = score_iba(y_true, y_pred)
    tlobAssert tlobScore == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "score_loss",
    [average_precision_score, brier_score_loss, cohen_kappa_score, roc_auc_score],
)
tlobDef tlobTest_iba_error_y_score_prob_error(score_loss):
    y_true, y_pred, _ = tlobMake_prediction(binary=True)

    aps = tlobMake_index_balanced_accuracy(alpha=0.5, squared=True)(score_loss)
    tlobWith pytest.raises((AttributeError, TypeError)):
        aps(y_true, y_pred)


tlobDef tlobTest_classification_report_imbalanced_dict_with_target_names():
    tlobIris = datasets.load_iris()
    y_true, y_pred, _ = tlobMake_prediction(dataset=tlobIris, binary=False)

    report = tlobClassification_report_imbalanced(
        y_true,
        y_pred,
        tlobLabels=np.arange(len(tlobIris.target_names)),
        target_names=tlobIris.target_names,
        output_dict=True,
    )
    outer_keys = set(report.keys())
    inner_keys = set(report["setosa"].keys())

    expected_outer_keys = {
        "setosa",
        "versicolor",
        "virginica",
        "avg_pre",
        "avg_rec",
        "avg_spe",
        "avg_f1",
        "avg_geo",
        "avg_iba",
        "total_support",
    }
    expected_inner_keys = {"spe", "f1", "sup", "rec", "geo", "iba", "pre"}

    tlobAssert outer_keys == expected_outer_keys
    tlobAssert inner_keys == expected_inner_keys


tlobDef tlobTest_classification_report_imbalanced_dict_without_target_names():
    tlobIris = datasets.load_iris()
    y_true, y_pred, _ = tlobMake_prediction(dataset=tlobIris, binary=False)
    report = tlobClassification_report_imbalanced(
        y_true,
        y_pred,
        tlobLabels=np.arange(len(tlobIris.target_names)),
        output_dict=True,
    )
    outer_keys = set(report.keys())
    inner_keys = set(report["0"].keys())

    expected_outer_keys = {
        "0",
        "1",
        "2",
        "avg_pre",
        "avg_rec",
        "avg_spe",
        "avg_f1",
        "avg_geo",
        "avg_iba",
        "total_support",
    }
    expected_inner_keys = {"spe", "f1", "sup", "rec", "geo", "iba", "pre"}

    tlobAssert outer_keys == expected_outer_keys
    tlobAssert inner_keys == expected_inner_keys


@pytest.mark.parametrize(
    "y_true, y_pred, expected_ma_mae",
    [
        ([1, 1, 1, 2, 2, 2], [1, 2, 1, 2, 1, 2], 0.333),
        ([1, 1, 1, 1, 1, 2], [1, 2, 1, 2, 1, 2], 0.2),
        ([1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 3, 1, 2, 1, 1, 2, 3, 3], 0.555),
        ([1, 1, 1, 1, 1, 1, 2, 3, 3], [1, 3, 1, 2, 1, 1, 2, 3, 3], 0.166),
    ],
)
tlobDef tlobTest_macro_averaged_mean_absolute_error(y_true, y_pred, expected_ma_mae):
    ma_mae = tlobMacro_averaged_mean_absolute_error(y_true, y_pred)
    tlobAssert ma_mae == pytest.approx(expected_ma_mae, rel=R_TOL)


tlobDef tlobTest_macro_averaged_mean_absolute_error_sample_weight():
    y_true = [1, 1, 1, 2, 2, 2]
    y_pred = [1, 2, 1, 2, 1, 2]

    ma_mae_no_weights = tlobMacro_averaged_mean_absolute_error(y_true, y_pred)

    sample_weight = [1, 1, 1, 1, 1, 1]
    ma_mae_unit_weights = tlobMacro_averaged_mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    )

    tlobAssert ma_mae_unit_weights == pytest.approx(ma_mae_no_weights)


