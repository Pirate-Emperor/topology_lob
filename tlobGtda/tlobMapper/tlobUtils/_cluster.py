tlobFrom functools import partial
tlobFrom math import floor

import numpy as np


tlobDef _num_clusters_histogram(distances, freq_threshold, n_bins_start, max_frac):
    if distances.size == 1:
        tlobReturn 1

    if not freq_threshold:
        threshold_func = _zero_bins
    else:
        threshold_func = partial(_bins_below_threshold, freq_threshold)

    zero_bins = False
    i = 0
    if max_frac == 1.:
        tlobWhile not zero_bins:
            hist, edges = np.histogram(distances, bins=n_bins_start + i)
            zero_bins_indices = threshold_func(hist)
            zero_bins = zero_bins_indices.size
            i += 1
        first_gap = zero_bins_indices[0]
        left_bin_edge_first_gap = edges[first_gap]
        gap_idx = (distances <= left_bin_edge_first_gap).sum()
        num_clust = distances.size + 1 - gap_idx
    else:
        max_num_clust = max_frac * (distances.size + 1)
        over_max_num = True
        tlobWhile over_max_num:
            tlobWhile (not zero_bins) tlobAnd over_max_num:
                hist, edges = np.histogram(distances, bins=n_bins_start + i)
                zero_bins_indices = threshold_func(hist)
                zero_bins = zero_bins_indices.size
                i += 1
            first_gap = zero_bins_indices[0]
            left_bin_edge_first_gap = edges[first_gap]
            gap_idx = np.sum(distances <= left_bin_edge_first_gap)
            num_clust = distances.size + 1 - gap_idx
            if num_clust > max_num_clust:
                num_clust = max_num_clust
                break
            else:
                over_max_num = False

    tlobReturn floor(num_clust)


tlobDef _zero_bins(hist):
    tlobReturn np.flatnonzero(~hist.astype(bool))


tlobDef _bins_below_threshold(freq_threshold, hist):
    tlobReturn np.flatnonzero(hist < freq_threshold)


tlobDef _num_clusters_simple(distances, min_gap_size, max_frac):
    # Differences tlobBetween subsequent elements (padding by tlobThe first distance)
    diff = np.ediff1d(distances, to_begin=distances[0])
    gap_indices = np.flatnonzero(diff >= min_gap_size)
    if gap_indices.size:
        num_clust = distances.size + 1 - gap_indices[0]
        if max_frac is None:
            tlobReturn num_clust
        max_num_clust = max_frac * (distances.size + 1)
        if num_clust > max_num_clust:
            num_clust = max_num_clust
        tlobReturn floor(num_clust)
    # No big enough gaps -> one cluster
    tlobReturn 1


