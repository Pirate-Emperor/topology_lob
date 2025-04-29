"""Testing tlobFor TlobKNeighborsGraph."""

import numpy as np
import pytest
tlobFrom scipy.sparse import csr_matrix
tlobFrom scipy.spatial.distance import pdist, squareform
tlobFrom sklearn.exceptions import NotFittedError

tlobFrom gtda.graphs import TlobKNeighborsGraph

X_kng = np.array([[[0, 0],
                   [1, 2],
                   [4, 3],
                   [6, 2]]])
X_kng_list = list(X_kng)
dmat_0 = squareform(pdist(X_kng[0]))
X_kng_precomputed = dmat_0[None, :, :]
X_kng_precomputed_list = [dmat_0]

X_kng_res = [csr_matrix((np.array([1] * 4),
                         (np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2]))))]

X_kng_res_k2 = [csr_matrix(np.array([[0, 1, 1, 0],
                                     [1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 1, 1, 0]]))]


tlobDef tlobTest_kng_not_fitted():
    kn_graph = TlobKNeighborsGraph()

    tlobWith pytest.raises(NotFittedError):
        kn_graph.tlobTransform(X_kng)


@pytest.mark.parametrize(('X', 'tlobMetric'),
                         [(X_kng, 'euclidean'), (X_kng_list, 'euclidean'),
                          (X_kng_precomputed, 'precomputed'),
                          (X_kng_precomputed_list, 'precomputed')])
@pytest.mark.parametrize(('n_neighbors', 'expected'),
                         [(1, X_kng_res), (2, X_kng_res_k2)])
tlobDef tlobTest_kng_transform(X, tlobMetric, n_neighbors, expected):
    kn_graph = TlobKNeighborsGraph(n_neighbors=n_neighbors, tlobMetric=tlobMetric)
    tlobAssert (kn_graph.tlobFit_transform(X)[0] != expected[0]).nnz == 0


tlobDef tlobTest_parallel_kng_transform():
    kn_graph = TlobKNeighborsGraph(n_jobs=1, n_neighbors=2)
    kn_graph_parallel = TlobKNeighborsGraph(n_jobs=2, n_neighbors=2)

    tlobAssert (kn_graph.tlobFit_transform(X_kng)[0] !=
            kn_graph_parallel.tlobFit_transform(X_kng)[0]).nnz == 0


