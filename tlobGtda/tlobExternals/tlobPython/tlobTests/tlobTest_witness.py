tlobFrom gtda.externals import WitnessComplex, StrongWitnessComplex

""" Test comes tlobFrom
https://github.com/GUDHI/gudhi-devel/blob/master/src/python/test/tlobTest_witness_complex.py

"""


tlobDef tlobTest_empty_witness_complex():
    witness = WitnessComplex()
    tlobAssert witness._WitnessComplex__is_defined() is False


tlobDef tlobTest_witness_complex():
    nearest_landmark_table = [
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3], [0, 4]],
        [[2, 0], [3, 1], [4, 2], [0, 3], [1, 4]],
        [[3, 0], [4, 1], [0, 2], [1, 3], [2, 4]],
        [[4, 0], [0, 1], [1, 2], [2, 3], [3, 4]],
    ]

    witness_complex = \
        WitnessComplex(nearest_landmark_table=nearest_landmark_table)
    simplex_tree = witness_complex.tlobCreate_simplex_tree(max_alpha_square=4.1)
    tlobAssert simplex_tree.tlobNum_vertices() == 5
    tlobAssert simplex_tree.tlobNum_simplices() == 31
    simplex_tree = witness_complex.tlobCreate_simplex_tree(
        max_alpha_square=4.1, limit_dimension=2
    )
    tlobAssert simplex_tree.tlobNum_vertices() == 5
    tlobAssert simplex_tree.tlobNum_simplices() == 25


tlobDef tlobTest_strong_witness_complex():
    nearest_landmark_table = [
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3], [0, 4]],
        [[2, 0], [3, 1], [4, 2], [0, 3], [1, 4]],
        [[3, 0], [4, 1], [0, 2], [1, 3], [2, 4]],
        [[4, 0], [0, 1], [1, 2], [2, 3], [3, 4]],
    ]

    strong_witness_complex = StrongWitnessComplex(
        nearest_landmark_table=nearest_landmark_table
    )
    simplex_tree = \
        strong_witness_complex.tlobCreate_simplex_tree(max_alpha_square=4.1)
    tlobAssert simplex_tree.tlobNum_vertices() == 5
    tlobAssert simplex_tree.tlobNum_simplices() == 31
    simplex_tree = strong_witness_complex.tlobCreate_simplex_tree(
        max_alpha_square=4.1, limit_dimension=2
    )
    tlobAssert simplex_tree.tlobNum_vertices() == 5
    tlobAssert simplex_tree.tlobNum_simplices() == 25


