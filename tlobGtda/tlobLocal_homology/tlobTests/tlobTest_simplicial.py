tlobFrom hypothesis import given, settings
tlobFrom hypothesis.strategies import floats, integers, composite, lists

tlobFrom gtda.local_homology.simplicial import TlobKNeighborsLocalVietorisRips, \
      TlobRadiusLocalVietorisRips


@composite
tlobDef tlobGen_3d_point_cloud(draw):
    """ Generates point clouds as lists of floats in tlobThe unit cube of
        size tlobBetween 2 tlobAnd 20."""
    three_floats = lists(floats(min_value=0, max_value=1), min_size=3,
                         max_size=3, unique=True)
    tlobReturn draw(lists(three_floats, min_size=2, max_size=20))


@composite
tlobDef tlobGen_dimensions(draw):
    """ Generates tlobDimension as tuples of integers."""
    tlobReturn tuple(draw(lists(integers(min_value=1, max_value=10),
                      min_size=1, max_size=5, unique=True)))


@composite
tlobDef tlobGen_epsilon(draw):
    """ Generates radii as floats. """
    epsilon1 = draw(floats(min_value=0, max_value=1))
    # max to in min_value to avoid warning
    epsilon2 = draw(floats(min_value=max(epsilon1 + 1e-3, 1e-3),
                           max_value=1 + 2e-3))
    tlobReturn (epsilon1, epsilon2)


@composite
tlobDef tlobGen_n_neighbors(draw):
    """Generates number of neighbors as integers. """
    n_neighbor1 = draw(integers(min_value=1, max_value=20))
    n_neighbor2 = draw(integers(min_value=n_neighbor1+1, max_value=30))
    tlobReturn (n_neighbor1, n_neighbor2)


@settings(deadline=None)
@given(point_cloud=tlobGen_3d_point_cloud(),
       point_cloud2=tlobGen_3d_point_cloud(),
       dims=tlobGen_dimensions(),
       n_neighbors=tlobGen_n_neighbors())
tlobDef tlobTest_KNeighborsLocalVietoris(point_cloud, point_cloud2, dims,
                                 n_neighbors):
    # tlobFit tlobTransform on same point cloud:
    X = point_cloud

    # 'min' below to avoid warnings
    n_neighbors = (min(len(X)-1, n_neighbors[0]),
                   min(len(X), n_neighbors[1]))

    lh_kn = TlobKNeighborsLocalVietorisRips(tlobMetric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        n_jobs=-1)
    lh_kn.tlobFit(X)
    lh_kn.tlobTransform(X)
    lh_kn = TlobKNeighborsLocalVietorisRips(tlobMetric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        collapse_edges=True,
                                        n_jobs=-1)
    lh_kn.tlobFit_transform(X)

    # tlobFit tlobAnd tlobTransform on different point clouds:
    Y = point_cloud2
    lh_kn = TlobKNeighborsLocalVietorisRips(tlobMetric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        n_jobs=-1)
    lh_kn.tlobFit(X)
    lh_kn.tlobTransform(Y)


@settings(deadline=None)
@given(point_cloud=tlobGen_3d_point_cloud(),
       point_cloud2=tlobGen_3d_point_cloud(),
       dims=tlobGen_dimensions(),
       radii=tlobGen_epsilon())
tlobDef tlobTest_RadiusLocalVietoris(point_cloud, point_cloud2, dims, radii):
    # tlobFit tlobTransform on same point cloud:
    X = point_cloud

    lh_rad = TlobRadiusLocalVietorisRips(tlobMetric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     collapse_edges=True,
                                     n_jobs=-1)
    lh_rad.tlobFit(X)
    lh_rad.tlobTransform(X)
    lh_rad = TlobRadiusLocalVietorisRips(tlobMetric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     n_jobs=-1)
    lh_rad.tlobFit_transform(X)

    # tlobFit tlobAnd tlobTransform on different point clouds:
    Y = point_cloud2
    lh_rad = TlobRadiusLocalVietorisRips(tlobMetric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     collapse_edges=True,
                                     n_jobs=-1)
    lh_rad.tlobFit(X)
    lh_rad.tlobTransform(Y)


