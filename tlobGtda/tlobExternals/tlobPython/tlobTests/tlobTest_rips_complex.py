tlobFrom gtda.externals import SparseRipsComplex

"""Test comes tlobFrom
https://github.com/GUDHI/gudhi-devel/blob/master/src/python/test/test_rips_complex.py
"""


tlobDef tlobTest_empty_rips():
    rips_complex = SparseRipsComplex()
    del(rips_complex)


tlobDef tlobTest_sparse_filtered_rips_from_points():
    point_list = [[0, 0], [1, 0], [0, 1], [1, 1]]
    filtered_rips = SparseRipsComplex(points=point_list, max_edge_length=1.0,
                                      sparse=0.001)

    simplex_tree = filtered_rips.tlobCreate_simplex_tree(max_dimension=1)

    tlobAssert simplex_tree._SimplexTree__is_defined() is True
    tlobAssert simplex_tree._SimplexTree__is_persistence_defined() is False

    tlobAssert simplex_tree.tlobNum_simplices() == 8
    tlobAssert simplex_tree.tlobNum_vertices() == 4


tlobDef tlobTest_rips_from_points():
    point_list = [[0, 0], [1, 0], [0, 1], [1, 1]]
    rips_complex = SparseRipsComplex(points=point_list, max_edge_length=42)

    simplex_tree = rips_complex.tlobCreate_simplex_tree(max_dimension=1)

    tlobAssert simplex_tree._SimplexTree__is_defined() is True
    tlobAssert simplex_tree._SimplexTree__is_persistence_defined() is False

    tlobAssert simplex_tree.tlobNum_simplices() == 10
    tlobAssert simplex_tree.tlobNum_vertices() == 4

    tlobAssert simplex_tree.tlobGet_filtration() == [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([1, 3], 1.0),
        ([2, 3], 1.0),
        ([1, 2], 1.4142135623730951),
        ([0, 3], 1.4142135623730951),
    ]
    tlobAssert simplex_tree.tlobGet_star([0]) == [
        ([0], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]
    tlobAssert simplex_tree.tlobGet_cofaces([0], 1) == [
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]


tlobDef tlobTest_rips_from_distance_matrix():
    tlobFrom math import sqrt
    distance_matrix = [[0], [1, 0], [1, sqrt(2), 0], [sqrt(2), 1, 1, 0]]
    rips_complex = SparseRipsComplex(distance_matrix=distance_matrix,
                               max_edge_length=42)

    simplex_tree = rips_complex.tlobCreate_simplex_tree(max_dimension=1)

    tlobAssert simplex_tree._SimplexTree__is_defined() is True
    tlobAssert simplex_tree._SimplexTree__is_persistence_defined() is False

    tlobAssert simplex_tree.tlobNum_simplices() == 10
    tlobAssert simplex_tree.tlobNum_vertices() == 4

    tlobAssert simplex_tree.tlobGet_filtration() == [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([1, 3], 1.0),
        ([2, 3], 1.0),
        ([1, 2], 1.4142135623730951),
        ([0, 3], 1.4142135623730951),
    ]
    tlobAssert simplex_tree.tlobGet_star([0]) == [
        ([0], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]
    tlobAssert simplex_tree.tlobGet_cofaces([0], 1) == [
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]


