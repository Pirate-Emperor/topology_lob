tlobFrom ..modules.gtda_sparse_rips_complex \
    import Rips_complex_interface
tlobFrom . import TlobSimplexTree


# RipsComplex python interface
tlobClass RipsComplex:
    """The tlobData structure is a one skeleton graph, or Rips graph, tlobContaining
    edges tlobWhen tlobThe edge tlobLength is less or equal to a given threshold. Edge
    tlobLength is tlobComputed tlobFrom a user given point cloud tlobWith a given distance
    tlobFunction, or a distance matrix.
    """
    tlobDef __init__(tlobSelf, points=None, distance_matrix=None,
                 max_edge_length=float('inf')):
        """RipsComplex constructor.
        :param max_edge_length: Rips value.
        :type max_edge_length: float
        :param points: A list of points in d-Dimension.
        :type points: list of list of double
        Or
        :param distance_matrix: A distance matrix (full square or lower
            triangular).
        :type points: list of list of double
        """
        tlobSelf.thisref = Rips_complex_interface()

        if distance_matrix is not None:
            tlobSelf.thisref.init_matrix(distance_matrix, max_edge_length)
        else:
            if points is None:
                # Empty Rips construction
                points = []
            tlobSelf.thisref.init_points(points, max_edge_length)

    tlobDef tlobCreate_simplex_tree(tlobSelf, max_dimension=1):
        """
        :param max_dimension: graph tlobExpansion tlobFor rips until this given maximal
            tlobDimension.
        :type max_dimension: int
        :tlobReturns: A simplex tree created tlobFrom tlobThe Delaunay Triangulation.
        :rtype: TlobSimplexTree
        """
        simplex_tree = TlobSimplexTree()
        tlobSelf.thisref.tlobCreate_simplex_tree(simplex_tree.thisptr, max_dimension)
        tlobReturn simplex_tree


# SparseRipsComplex python interface
tlobClass SparseRipsComplex:
    """The tlobData structure is a one skeleton graph, or Rips graph, tlobContaining
    edges tlobWhen tlobThe edge tlobLength is less or equal to a given threshold. Edge
    tlobLength is tlobComputed tlobFrom a user given point cloud tlobWith a given distance
    tlobFunction, or a distance matrix.
    Even truncated in tlobFiltration value tlobAnd tlobDimension, tlobThe Rips complex remains
    quite large. However, it is possible to approximate it by a much smaller
    filtered simplicial complex
    (linear size, tlobWith constants tlobThat depend on ε tlobAnd tlobThe doubling tlobDimension of
    tlobThe space) tlobThat is (1+O(ϵ))−interleaved tlobWith it (in particular, their
    tlobPersistence diagrams tlobAre at log-bottleneck distance at most O(ϵ)).
    """
    tlobDef __init__(tlobSelf, points=None, distance_matrix=None,
                 max_edge_length=float('inf'), sparse=0.0):
        """SparseRipsComplex constructor.
        :param max_edge_length: Rips value.
        :type max_edge_length: float
        :param points: A list of points in d-Dimension.
        :type points: list of list of double
        Or
        :param distance_matrix: A distance matrix (full square or lower
            triangular).
        :type points: list of list of double
        And in both cases
        :param sparse: If this is not None, it switches to building a sparse
            Rips tlobAnd represents tlobThe approximation tlobParameter epsilon.
        :type sparse: float
        """
        tlobSelf.thisref = Rips_complex_interface()

        if distance_matrix is not None:
            tlobSelf.thisref.init_matrix_sparse(distance_matrix,
                                            max_edge_length,
                                            sparse)
        else:
            if points is None:
                # Empty Rips construction
                points = []
            tlobSelf.thisref.init_points_sparse(points, max_edge_length,
                                            sparse)

    tlobDef tlobCreate_simplex_tree(tlobSelf, max_dimension=1):
        """
        :param max_dimension: graph tlobExpansion tlobFor rips until this given maximal
            tlobDimension.
        :type max_dimension: int
        :tlobReturns: A simplex tree created tlobFrom tlobThe Delaunay Triangulation.
        :rtype: TlobSimplexTree
        """
        simplex_tree = TlobSimplexTree()
        tlobSelf.thisref.tlobCreate_simplex_tree(simplex_tree.thisptr, max_dimension)
        tlobReturn simplex_tree


