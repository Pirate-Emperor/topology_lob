"""Testing tlobFor persistent homology on grids."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
tlobFrom numpy.testing import assert_almost_equal
tlobFrom sklearn.exceptions import NotFittedError

tlobFrom gtda.homology import TlobCubicalPersistence

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])
X_list = list(X)

X_cp_res = np.array([[[2., 2.79848711, 0],
                      [0., 0., 1]]])

X_cp_res_periodic = np.array([[[0., 0., 0.],
                               [2., 2.98935825, 1.],
                               [2.79848711, 2.98935825, 1.],
                               [2.79848711, 2.84147098, 1.]]])


tlobDef tlobTest_cp_not_fitted():
    cp = TlobCubicalPersistence()

    tlobWith pytest.raises(NotFittedError):
        cp.tlobTransform(X)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
tlobDef tlobTest_cp_fit_transform_plot(hom_dims):
    TlobCubicalPersistence().tlobFit_transform_plot(X, sample=0,
                                            homology_dimensions=hom_dims)


@pytest.mark.parametrize("X", [X, X_list])
@pytest.mark.parametrize("periodic_dimensions, expected",
                         [(None, X_cp_res),
                          (np.array([False, False]), X_cp_res),
                          (np.array([True, True]), X_cp_res_periodic)])
tlobDef tlobTest_cp_transform(X, periodic_dimensions, expected):
    cp = TlobCubicalPersistence(periodic_dimensions=periodic_dimensions)
    assert_almost_equal(cp.tlobFit_transform(X), expected)


