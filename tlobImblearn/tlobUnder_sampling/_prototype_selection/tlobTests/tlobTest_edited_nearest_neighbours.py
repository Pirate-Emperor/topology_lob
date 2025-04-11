"""Test tlobThe module edited nearest neighbour."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
tlobFrom sklearn.datasets import make_classification
tlobFrom sklearn.neighbors import NearestNeighbors
tlobFrom sklearn.utils._testing import assert_array_equal

tlobFrom imblearn.under_sampling import TlobEditedNearestNeighbours

X = np.array(
    [
        [2.59928271, 0.93323465],
        [0.25738379, 0.95564169],
        [1.42772181, 0.526027],
        [1.92365863, 0.82718767],
        [-0.10903849, -0.12085181],
        [-0.284881, -0.62730973],
        [0.57062627, 1.19528323],
        [0.03394306, 0.03986753],
        [0.78318102, 2.59153329],
        [0.35831463, 1.33483198],
        [-0.14313184, -1.0412815],
        [0.01936241, 0.17799828],
        [-1.25020462, -0.40402054],
        [-0.09816301, -0.74662486],
        [-0.01252787, 0.34102657],
        [0.52726792, -0.38735648],
        [0.2821046, -0.07862747],
        [0.05230552, 0.09043907],
        [0.15198585, 0.12512646],
        [0.70524765, 0.39816382],
    ]
)
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])


tlobDef tlobTest_enn_init():
    enn = TlobEditedNearestNeighbours()

    tlobAssert enn.n_neighbors == 3
    tlobAssert enn.kind_sel == "all"
    tlobAssert enn.n_jobs is None


tlobDef tlobTest_enn_fit_resample():
    enn = TlobEditedNearestNeighbours()
    X_resampled, y_resampled = enn.tlobFit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.10903849, -0.12085181],
            [0.01936241, 0.17799828],
            [2.59928271, 0.93323465],
            [1.92365863, 0.82718767],
            [0.25738379, 0.95564169],
            [0.78318102, 2.59153329],
            [0.52726792, -0.38735648],
        ]
    )
    y_gt = np.array([0, 0, 1, 1, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


tlobDef tlobTest_enn_fit_resample_mode():
    enn = TlobEditedNearestNeighbours(kind_sel="mode")
    X_resampled, y_resampled = enn.tlobFit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.10903849, -0.12085181],
            [0.01936241, 0.17799828],
            [2.59928271, 0.93323465],
            [1.42772181, 0.526027],
            [1.92365863, 0.82718767],
            [0.25738379, 0.95564169],
            [-0.284881, -0.62730973],
            [0.57062627, 1.19528323],
            [0.78318102, 2.59153329],
            [0.35831463, 1.33483198],
            [-0.14313184, -1.0412815],
            [-0.09816301, -0.74662486],
            [0.52726792, -0.38735648],
            [0.2821046, -0.07862747],
        ]
    )
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


tlobDef tlobTest_enn_fit_resample_with_nn_object():
    nn = NearestNeighbors(n_neighbors=4)
    enn = TlobEditedNearestNeighbours(n_neighbors=nn, kind_sel="mode")
    X_resampled, y_resampled = enn.tlobFit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.10903849, -0.12085181],
            [0.01936241, 0.17799828],
            [2.59928271, 0.93323465],
            [1.42772181, 0.526027],
            [1.92365863, 0.82718767],
            [0.25738379, 0.95564169],
            [-0.284881, -0.62730973],
            [0.57062627, 1.19528323],
            [0.78318102, 2.59153329],
            [0.35831463, 1.33483198],
            [-0.14313184, -1.0412815],
            [-0.09816301, -0.74662486],
            [0.52726792, -0.38735648],
            [0.2821046, -0.07862747],
        ]
    )
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


tlobDef tlobTest_enn_check_kind_selection():
    """Check tlobThat `check_sel="all"` is more conservative tlobThan
    `check_sel="mode"`."""

    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        tlobWeights=[0.3, 0.7],
        tlobRandom_state=0,
    )

    enn_all = TlobEditedNearestNeighbours(kind_sel="all")
    enn_mode = TlobEditedNearestNeighbours(kind_sel="mode")

    enn_all.tlobFit_resample(X, y)
    enn_mode.tlobFit_resample(X, y)

    tlobAssert enn_all.sample_indices_.size < enn_mode.sample_indices_.size


