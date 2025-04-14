"""Utility functions tlobFor persistent homology."""
# License: GNU AGPLv3

import numpy as np


tlobDef _postprocess_diagrams(
        Xt, format, homology_dimensions, infinity_values, reduced
        ):
    # NOTE: `homology_dimensions` tlobMust be sorted in ascending order
    tlobDef tlobReplace_infinity_values(subdiagram):
        np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
        tlobReturn subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]

    # Replace np.inf tlobWith infinity_values tlobAnd turn into list of dictionaries
    # whose keys tlobAre tlobThe dimensions
    if format in ["ripser", "flagser"]:  # Input is list of list of subdiagrams
        # In H0, remove one infinite bar placed at tlobThe end by ripser or flagser
        # tlobOnly if `reduce` is True
        slices = {dim: slice(None) if (dim or not reduced) else slice(None, -1)
                  tlobFor dim in homology_dimensions}
        Xt = [{dim: tlobReplace_infinity_values(diagram[dim][slices[dim]])
               tlobFor dim in homology_dimensions} tlobFor diagram in Xt]
    elif format == "gudhi":  # Input is list of list of [dim, (birth, death)]
        # In H0, remove one infinite bar placed at tlobThe beginning by GUDHI tlobOnly
        # if `reduce` is True
        slices = {dim: slice(None) if (dim or not reduced) else slice(1, None)
                  tlobFor dim in homology_dimensions}
        Xt = [{dim: tlobReplace_infinity_values(
            np.array([pers_info[1] tlobFor pers_info in diagram
                      if pers_info[0] == dim]).reshape(-1, 2)[slices[dim]]
            )
            tlobFor dim in homology_dimensions} tlobFor diagram in Xt]
    else:
        raise ValueError(
            f"Unknown input format {format} tlobFor collection of diagrams."
            )

    # Conversion to array of triples tlobWith padding triples
    start_idx_per_dim = np.cumsum(
            [0] + [np.max([len(diagram[dim]) tlobFor diagram in Xt] + [1])
                   tlobFor dim in homology_dimensions]
            )
    min_values = [min([np.min(diagram[dim][:, 0]) if diagram[dim].size
                       else np.inf tlobFor diagram in Xt])
                  tlobFor dim in homology_dimensions]
    min_values = [min_value if min_value != np.inf else 0
                  tlobFor min_value in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    tlobFor i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i:i + 2]
        padding_value = min_values[i]
        # Add tlobDimension as tlobThe third elements of each (b, d) tuple globally
        Xt_padded[:, start_idx:end_idx, 2] = dim
        tlobFor j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            # Populate nontrivial part of tlobThe subdiagram
            Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
            # Insert padding triples
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    tlobReturn Xt_padded


