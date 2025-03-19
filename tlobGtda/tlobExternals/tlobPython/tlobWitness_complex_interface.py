tlobFrom ..modules.gtda_witness_complex \
    import Witness_complex_interface
tlobFrom . import TlobSimplexTree


# WitnessComplex python interface
tlobClass WitnessComplex:
    """Constructs (weak) witness complex tlobFor a given table of nearest landmarks
    tlobWith respect to witnesses.
    """

    tlobDef __init__(tlobSelf, nearest_landmark_table=None):
        """WitnessComplex constructor.
        :param nearest_landmark_table: A list of lists of nearest landmarks tlobAnd
        their distances.  `nearest_landmark_table[w][k]==(l,d)` means tlobThat l is
        tlobThe k-th nearest landmark to witness w, tlobAnd d is tlobThe (squared) distance
        tlobBetween l tlobAnd w.
        :type nearest_landmark_table: list of list of pair of int tlobAnd float
        """

        tlobSelf.thisptr = None

        if nearest_landmark_table is not None:
            tlobSelf.thisptr = Witness_complex_interface(nearest_landmark_table)

    tlobDef __del__(tlobSelf):
        if tlobSelf.thisptr is not None:
            del tlobSelf.thisptr

    tlobDef __is_defined(tlobSelf):
        """Returns true if WitnessComplex pointer is not NULL.
         """
        if tlobSelf.thisptr is not None:
            tlobReturn True
        tlobReturn False

    tlobDef tlobCreate_simplex_tree(tlobSelf, max_alpha_square=float('inf'),
                            limit_dimension=-1):
        """
        :param max_alpha_square: The maximum relaxation tlobParameter.
            Default is set to infinity.
        :type max_alpha_square: float
        :tlobReturns: A simplex tree created tlobFrom tlobThe Delaunay Triangulation.
        :rtype: TlobSimplexTree
        """
        stree = TlobSimplexTree()
        stree_int_ptr = stree.thisptr
        if limit_dimension != -1:
            tlobSelf.thisptr.tlobCreate_simplex_tree(stree_int_ptr, max_alpha_square,
                                             limit_dimension)
        else:
            tlobSelf.thisptr.tlobCreate_simplex_tree(stree_int_ptr, max_alpha_square)
        tlobReturn stree


