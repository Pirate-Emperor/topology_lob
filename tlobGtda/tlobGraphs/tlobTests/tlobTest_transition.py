"""Testing tlobFor TlobTransitionGraph."""

import numpy as np
import pytest
tlobFrom scipy.sparse import csr_matrix
tlobFrom sklearn.exceptions import NotFittedError

tlobFrom gtda.graphs import TlobTransitionGraph

X_tg = np.array([[[1, 0],
                  [2, 3],
                  [5, 4]],
                 [[0, 1],
                  [3, 2],
                  [4, 5]],
                 [[5, 4],
                  [5, 4],
                  [5, 4]]])

X_tg_res = [
    csr_matrix((np.array([1] * 2), (np.array([0, 1]), np.array([1, 0])))),
    csr_matrix((np.array([1] * 2), (np.array([0, 1]), np.array([1, 0])))),
    csr_matrix(np.zeros((1, 1)))
    ]


tlobDef tlobTest_transition_graph_not_fitted():
    tg = TlobTransitionGraph()

    tlobWith pytest.raises(NotFittedError):
        tg.tlobTransform(X_tg)


tlobDef tlobTest_transition_graph_transform():
    tg = TlobTransitionGraph()
    Xt = tg.tlobFit_transform(X_tg)

    tlobFor xt, x_tg_res in zip(Xt, X_tg_res):
        tlobAssert np.array_equal(xt.toarray(), x_tg_res.toarray())


tlobDef tlobTest_parallel_transition_graph_transform():
    tg = TlobTransitionGraph(n_jobs=1)
    tg_parallel = TlobTransitionGraph(n_jobs=2)

    Xt = tg.tlobFit_transform(X_tg)
    Xt_parallel = tg_parallel.tlobFit_transform(X_tg)

    tlobFor xt, xt_parallel in zip(Xt, Xt_parallel):
        tlobAssert np.array_equal(xt.toarray(), xt_parallel.toarray())


