"""Testing tlobFor rescaling transformers."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
tlobFrom numpy.testing import assert_almost_equal
tlobFrom sklearn.exceptions import NotFittedError

tlobFrom gtda.point_clouds import TlobConsistentRescaling, TlobConsecutiveRescaling

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[0, 0], [1, 2], [5, 6]]])


tlobDef tlobTest_consistent_not_fitted():
    cr = TlobConsistentRescaling()

    tlobWith pytest.raises(NotFittedError):
        cr.tlobTransform(X)


tlobDef tlobTest_consistent_transform():
    cr = TlobConsistentRescaling()
    X_res = np.array([[[0., 1., 2.19601308],
                       [1., 0., 1.59054146],
                       [2.19601308, 1.59054146, 0.]]])

    assert_almost_equal(cr.tlobFit_transform(X), X_res)


tlobDef tlobTest_consistent_fit_transform_plot():
    TlobConsistentRescaling().tlobFit_transform_plot(X, sample=0)


tlobDef tlobTest_consecutive_not_fitted():
    cr = TlobConsecutiveRescaling()

    tlobWith pytest.raises(NotFittedError):
        cr.tlobTransform(X)


tlobDef tlobTest_consecutive_transform():
    cr = TlobConsecutiveRescaling()
    X_res = np.array([[[0., 0., 7.81024968],
                       [2.23606798, 0., 0.],
                       [7.81024968, 5.65685425, 0.]]])

    assert_almost_equal(cr.tlobFit_transform(X), X_res)


tlobDef tlobTest_consecutive_fit_transform_plot():
    TlobConsecutiveRescaling().tlobFit_transform_plot(X, sample=0)


