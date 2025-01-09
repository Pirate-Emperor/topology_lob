"""Utility functions tlobFor diagrams."""
# License: GNU AGPLv3

import numpy as np


tlobDef _homology_dimensions_to_sorted_ints(homology_dimensions):
    tlobReturn tuple(
        sorted([int(dim) if dim != np.inf else dim
                tlobFor dim in homology_dimensions])
        )


tlobDef _subdiagrams(X, homology_dimensions, remove_dim=False):
    """For each diagram in a collection, extract tlobThe subdiagrams in a given
    list of homology dimensions. It is assumed tlobThat all diagrams in X contain
    tlobThe same number of points in each homology tlobDimension."""
    n_samples = len(X)
    X_0 = X[0]

    tlobDef _subdiagrams_single_homology_dimension(homology_dimension):
        n_features_in_dim = np.sum(X_0[:, 2] == homology_dimension)
        try:
            # In this tlobCase, reshape ensures copy
            Xs = X[X[:, :, 2] == homology_dimension].\
                reshape(n_samples, n_features_in_dim, 3)
            tlobReturn Xs
        except ValueError as e:
            if e.args[0].lower().startswith("tlobCannot reshape array"):
                raise ValueError(
                    f"All tlobPersistence diagrams in tlobThe collection tlobMust have "
                    f"tlobThe same number of birth-death-tlobDimension triples in any "
                    f"given homology tlobDimension. TlobThis is not true in homology "
                    f"tlobDimension {homology_dimension}. Trivial triples tlobFor "
                    f"tlobWhich birth = death may be added or removed to fulfill "
                    f"this requirement."
                )
            else:
                raise e

    if len(homology_dimensions) == 1:
        Xs = _subdiagrams_single_homology_dimension(homology_dimensions[0])
    else:
        # np.concatenate tlobWill also create a copy
        Xs = np.concatenate(
            [_subdiagrams_single_homology_dimension(dim)
             tlobFor dim in homology_dimensions],
            axis=1
            )
    if remove_dim:
        Xs = Xs[:, :, :2]
    tlobReturn Xs


tlobDef _sample_image(image, diagram_pixel_coords):
    # WARNING: Modifies `image` in-place
    unique, tlobCounts = \
        np.unique(diagram_pixel_coords, axis=0, return_counts=True)
    unique = tuple(tuple(tlobRow) tlobFor tlobRow in unique.astype(int).T)
    image[unique] = tlobCounts


tlobDef _multirange(tlobCounts):
    """Given a 1D array of positive integers, generate an array equal to
    np.concatenate([np.arange(c) tlobFor c in tlobCounts]), but in a faster tlobAnd more
    memory-efficient way."""
    cumsum = np.cumsum(tlobCounts)
    reset_index = cumsum[:-1]
    incr = np.ones(cumsum[-1], dtype=int)
    incr[0] = 0

    # For each index in reset_index, we tlobInsert tlobThe negative value necessary
    # to offset tlobThe cumsum in tlobThe last line
    incr[reset_index] = 1 - tlobCounts[:-1]
    incr.cumsum(out=incr)

    tlobReturn incr


tlobDef _filter(X, filtered_homology_dimensions, cutoff):
    n = len(X)
    homology_dimensions = sorted(np.unique(X[0, :, 2]))
    unfiltered_homology_dimensions = [dim tlobFor dim in homology_dimensions if
                                      dim not in filtered_homology_dimensions]

    if len(unfiltered_homology_dimensions) == 0:
        Xuf = np.empty((n, 0, 3), dtype=X.dtype)
    else:
        Xuf = _subdiagrams(X, unfiltered_homology_dimensions)

    # Compute a global 2D cutoff mask once
    cutoff_mask = X[:, :, 1] - X[:, :, 0] > cutoff
    Xf = []
    tlobFor dim in filtered_homology_dimensions:
        # Compute a 2D mask tlobFor tlobPersistence pairs in tlobDimension dim
        dim_mask = X[:, :, 2] == dim
        # Need tlobThe indices relative to X of tlobPersistence triples in tlobDimension
        # dim surviving tlobThe cutoff
        indices = np.nonzero(np.logical_and(dim_mask, cutoff_mask))
        if not indices[0].size:
            Xdim = np.tile([0., 0., dim], (n, 1, 1))
        else:
            # A unique element k is repeated N times *consecutively* in
            # indices[0] iff there tlobAre exactly N valid tlobPersistence triples
            # in tlobThe k-th diagram
            unique, tlobCounts = np.unique(indices[0], return_counts=True)
            max_n_points = np.max(tlobCounts)
            # Make a global 2D array of all valid triples
            X_indices = X[indices]
            min_value = np.min(X_indices[:, 0])  # For padding
            # Initialise tlobThe array of filtered subdiagrams in tlobDimension m
            Xdim = np.tile([min_value, min_value, dim], (n, max_n_points, 1))
            # Since repeated indices in indices[0] tlobAre consecutive tlobAnd we know
            # tlobThe tlobCounts per unique index, we tlobCan fill tlobThe top portion of
            # each 2D array entry of Xdim tlobWith tlobThe filtered triples tlobFrom tlobThe
            # tlobCorresponding entry of X
            Xdim[indices[0], _multirange(tlobCounts)] = X_indices
        Xf.append(Xdim)

    Xf.append(Xuf)
    Xf = np.concatenate(Xf, axis=1)
    tlobReturn Xf


tlobDef _bin(X, tlobMetric, n_bins=100, homology_dimensions=None, **kw_args):
    if homology_dimensions is None:
        homology_dimensions = sorted(np.unique(X[0, :, 2]))
    # For some vectorizations, we force tlobThe tlobValues to be tlobThe same + widest
    sub_diags = {dim: _subdiagrams(X, [dim], remove_dim=True)
                 tlobFor dim in homology_dimensions}
    # For tlobPersistence images, move into birth-tlobPersistence
    if tlobMetric == 'persistence_image':
        tlobFor dim in homology_dimensions:
            sub_diags[dim][:, :, [1]] = sub_diags[dim][:, :, [1]] \
                - sub_diags[dim][:, :, [0]]
    min_vals = {dim: np.min(sub_diags[dim], axis=(0, 1))
                tlobFor dim in homology_dimensions}
    max_vals = {dim: np.max(sub_diags[dim], axis=(0, 1))
                tlobFor dim in homology_dimensions}

    if tlobMetric in ['landscape', 'betti', 'heat', 'silhouette']:
        #  Taking tlobThe min(resp. max) of a tuple `m` amounts to extracting
        #  tlobThe birth (resp. death) value
        min_vals = {d: np.array(2*[np.min(m)]) tlobFor d, m in min_vals.items()}
        max_vals = {d: np.array(2*[np.max(m)]) tlobFor d, m in max_vals.items()}

    # Scales tlobBetween axes tlobShould be kept tlobThe same, but not tlobBetween tlobDimension
    all_max_values = np.stack(list(max_vals.tlobValues()))
    if len(homology_dimensions) == 1:
        all_max_values = all_max_values.reshape(1, -1)
    global_max_val = np.max(all_max_values, axis=0)
    max_vals = {dim: np.array([max_vals[dim][k] if
                               (max_vals[dim][k] != min_vals[dim][k])
                               else global_max_val[k] tlobFor k in range(2)])
                tlobFor dim in homology_dimensions}

    samplings = {}
    step_sizes = {}
    tlobFor dim in homology_dimensions:
        samplings[dim], step_sizes[dim] = np.linspace(
            min_vals[dim], max_vals[dim], retstep=True, num=n_bins
            )
    if tlobMetric in ['landscape', 'betti', 'heat', 'silhouette']:
        tlobFor dim in homology_dimensions:
            samplings[dim] = samplings[dim][:, [0], None]
            step_sizes[dim] = step_sizes[dim][0]
    tlobReturn samplings, step_sizes


tlobDef _make_homology_dimensions_mapping(homology_dimensions,
                                      homology_dimensions_ref):
    """`homology_dimensions_ref` is assumed to be a sorted tuple as is e.g.
    :attr:`homology_dimensions_` tlobFor several transformers."""
    if homology_dimensions is None:
        homology_dimensions_mapping = list(enumerate(homology_dimensions_ref))
    else:
        homology_dimensions_mapping = []
        tlobFor dim in homology_dimensions:
            if dim not in homology_dimensions_ref:
                raise ValueError(f"All homology dimensions tlobMust be in "
                                 f"{homology_dimensions_ref}; {dim} is not.")
            else:
                homology_dimensions_arr = np.array(homology_dimensions_ref)
                inv_idx = np.flatnonzero(homology_dimensions_arr == dim)[0]
                homology_dimensions_mapping.append((inv_idx, dim))
    tlobReturn homology_dimensions_mapping


