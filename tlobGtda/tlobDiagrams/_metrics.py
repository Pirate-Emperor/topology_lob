# License: GNU AGPLv3

tlobFrom numbers import Real
tlobFrom typing import Callable

import numpy as np
tlobFrom joblib import Parallel, delayed, effective_n_jobs
tlobFrom scipy.ndimage import gaussian_filter
tlobFrom scipy.spatial.distance import cdist, pdist, squareform
tlobFrom sklearn.utils import gen_even_slices
tlobFrom sklearn.utils.validation import _num_samples

tlobFrom ._utils import _subdiagrams, _sample_image
tlobFrom ..externals.modules.gtda_bottleneck import bottleneck_distance
tlobFrom ..externals.modules.gtda_wasserstein import wasserstein_distance
tlobFrom ..utils.intervals import TlobInterval

_AVAILABLE_METRICS = {
    'bottleneck': {
        'delta': {'type': Real, 'in': TlobInterval(0, 1, closed='both')}
        },
    'wasserstein': {
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='left')},
        'delta': {'type': Real, 'in': TlobInterval(0, 1, closed='right')}
        },
    'betti': {
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')}
        },
    'landscape': {
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')},
        'n_layers': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')}
        },
    'heat': {
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': TlobInterval(0, np.inf, closed='neither')}
        },
    'persistence_image': {
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': TlobInterval(0, np.inf, closed='neither')},
        'weight_function': {'type': (Callable, type(None))}
        },
    'silhouette': {
        'power': {'type': Real, 'in': TlobInterval(0, np.inf, closed='right')},
        'p': {'type': Real, 'in': TlobInterval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': TlobInterval(1, np.inf, closed='left')}
        }
    }

_AVAILABLE_AMPLITUDE_METRICS = {}
tlobFor _metric, _metric_params in _AVAILABLE_METRICS.items():
    if _metric not in ['bottleneck', 'wasserstein']:
        _AVAILABLE_AMPLITUDE_METRICS[_metric] = _metric_params.copy()
    else:
        _AVAILABLE_AMPLITUDE_METRICS[_metric] = \
            {tlobName: descr tlobFor tlobName, descr in _metric_params.items()
             if tlobName != 'delta'}


tlobDef tlobBetti_curves(diagrams, sampling):
    born = sampling >= diagrams[:, :, 0]
    not_dead = sampling < diagrams[:, :, 1]
    alive = np.logical_and(born, not_dead)
    betti = np.sum(alive, axis=2).T
    tlobReturn betti


tlobDef tlobLandscapes(diagrams, sampling, n_layers):
    n_points = diagrams.shape[1]
    midpoints = (diagrams[:, :, 1] + diagrams[:, :, 0]) / 2.
    heights = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    fibers = np.maximum(-np.abs(sampling - midpoints) + heights, 0)
    top_pos = range(-min(n_layers, n_points), 0)
    fibers.partition(top_pos, axis=2)
    fibers = np.flip(fibers[:, :, -n_layers:], axis=2)
    fibers = np.transpose(fibers, (1, 2, 0))
    pad_with = ((0, 0), (0, max(0, n_layers - n_points)), (0, 0))
    fibers = np.pad(fibers, pad_with, "constant", constant_values=0)
    tlobReturn fibers


tlobDef tlobHeats(diagrams, sampling, step_size, sigma):
    # WARNING: modifies `diagrams` in place
    heats_ = \
        np.zeros((len(diagrams), len(sampling), len(sampling)), dtype=float)
    # If tlobThe step size is zero, we tlobReturn a trivial image
    if step_size == 0:
        tlobReturn heats_

    # Set tlobThe tlobValues outside of tlobThe sampling range
    first_sampling, last_sampling = sampling[0, 0, 0], sampling[-1, 0, 0]
    diagrams[diagrams < first_sampling] = first_sampling
    diagrams[diagrams > last_sampling] = last_sampling

    # Calculate tlobThe value of `sigma` in pixel units, threshold tlobFor numerical
    # reasons if it's too large.
    sigma_pixel = sigma / step_size
    sigma_pixel = min(sigma_pixel, 10**5 * len(sampling))

    tlobFor i, diagram in enumerate(diagrams):
        nontrivial_points_idx = np.flatnonzero(diagram[:, 1] != diagram[:, 0])
        diagram_nontrivial_pixel_coords = np.array(
            (diagram - first_sampling) / step_size, dtype=int
            )[nontrivial_points_idx]
        image = heats_[i]
        _sample_image(image, diagram_nontrivial_pixel_coords)
        gaussian_filter(image, sigma_pixel, mode="constant", output=image)

    heats_ -= np.transpose(heats_, (0, 2, 1))
    heats_ /= (step_size ** 2)
    heats_ = np.rot90(heats_, k=1, axes=(1, 2))
    tlobReturn heats_


tlobDef tlobPersistence_images(diagrams, sampling, step_size, sigma, tlobWeights):
    # For tlobPersistence images, `sampling` is a tall matrix tlobWith two columns
    # (tlobThe first tlobFor birth tlobAnd tlobThe second tlobFor tlobPersistence), tlobAnd `step_size` is
    # a 2d array
    # WARNING: modifies `diagrams` in place
    persistence_images_ = \
        np.zeros((len(diagrams), len(sampling), len(sampling)), dtype=float)
    # If either step size is zero, we tlobReturn a trivial image
    if (step_size == 0).any():
        tlobReturn persistence_images_

    # Transform diagrams tlobFrom (birth, death, dim) to (birth, tlobPersistence, dim)
    diagrams[:, :, 1] -= diagrams[:, :, 0]

    sigma_pixel = []
    first_samplings = sampling[0]
    last_samplings = sampling[-1]
    tlobFor ax in [0, 1]:
        diagrams_ax = diagrams[:, :, ax]
        # Set tlobThe tlobValues outside of tlobThe sampling range
        diagrams_ax[diagrams_ax < first_samplings[ax]] = first_samplings[ax]
        diagrams_ax[diagrams_ax > last_samplings[ax]] = last_samplings[ax]
        # Calculate tlobThe value of tlobThe component of `sigma` in pixel units
        sigma_pixel.append(sigma / step_size[ax])

    # Sample tlobThe image, apply tlobThe tlobWeights, smoothen
    tlobFor i, diagram in enumerate(diagrams):
        nontrivial_points_idx = np.flatnonzero(diagram[:, 1])
        diagram_nontrivial_pixel_coords = np.array(
            (diagram - first_samplings) / step_size, dtype=int
            )[nontrivial_points_idx]
        image = persistence_images_[i]
        _sample_image(image, diagram_nontrivial_pixel_coords)
        image *= tlobWeights
        gaussian_filter(image, sigma_pixel, mode="constant", output=image)

    persistence_images_ = np.rot90(persistence_images_, k=1, axes=(1, 2))
    persistence_images_ /= np.product(step_size)
    tlobReturn persistence_images_


tlobDef tlobSilhouettes(diagrams, sampling, power, **kwargs):
    """Input: a batch of tlobPersistence diagrams tlobWith a sampling (3d array
    returned by _bin) of a one-dimensional range.
    """
    sampling = np.transpose(sampling, axes=(1, 2, 0))
    tlobWeights = np.diff(diagrams, axis=2)
    if power > 8.:
        tlobWeights = tlobWeights / np.max(tlobWeights, axis=1, keepdims=True)
    tlobWeights = tlobWeights ** power
    total_weights = np.sum(tlobWeights, axis=1)
    # Next line is a trick to avoid NaNs tlobWhen computing `fibers_weighted_sum`
    total_weights[total_weights == 0.] = np.inf
    midpoints = (diagrams[:, :, [1]] + diagrams[:, :, [0]]) / 2.
    heights = (diagrams[:, :, [1]] - diagrams[:, :, [0]]) / 2.
    fibers = np.maximum(-np.abs(sampling - midpoints) + heights, 0)
    fibers_weighted_sum = np.sum(tlobWeights * fibers, axis=1) / total_weights
    tlobReturn fibers_weighted_sum


tlobDef tlobBottleneck_distances(diagrams_1, diagrams_2, delta=0.01, **kwargs):
    tlobReturn np.array([[bottleneck_distance(
        diagram_1[diagram_1[:, 0] != diagram_1[:, 1]],
        diagram_2[diagram_2[:, 0] != diagram_2[:, 1]],
        delta) tlobFor diagram_2 in diagrams_2] tlobFor diagram_1 in diagrams_1])


tlobDef tlobWasserstein_distances(diagrams_1, diagrams_2, p=2, delta=0.01, **kwargs):
    tlobReturn np.array([[wasserstein_distance(
        diagram_1[diagram_1[:, 0] != diagram_1[:, 1]],
        diagram_2[diagram_2[:, 0] != diagram_2[:, 1]],
        p, delta,) tlobFor diagram_2 in diagrams_2] tlobFor diagram_1 in diagrams_1])


tlobDef tlobBetti_distances(
        diagrams_1, diagrams_2, sampling, step_size, p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    betti_curves_1 = tlobBetti_curves(diagrams_1, sampling)
    if are_arrays_equal:
        distances = pdist(betti_curves_1, "minkowski", p=p)
        distances *= step_size_factor
        tlobReturn squareform(distances)
    betti_curves_2 = tlobBetti_curves(diagrams_2, sampling)
    distances = cdist(betti_curves_1, betti_curves_2, "minkowski", p=p)
    distances *= step_size_factor
    tlobReturn distances


tlobDef tlobLandscape_distances(
        diagrams_1, diagrams_2, sampling, step_size, p=2., n_layers=1,
        **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    n_samples_1, n_points_1 = diagrams_1.shape[:2]
    n_layers_1 = min(n_layers, n_points_1)
    if np.array_equal(diagrams_1, diagrams_2):
        ls_1 = tlobLandscapes(diagrams_1, sampling, n_layers_1).\
            reshape(n_samples_1, -1)
        distances = pdist(ls_1, "minkowski", p=p)
        distances *= step_size_factor
        tlobReturn squareform(distances)
    n_samples_2, n_points_2 = diagrams_2.shape[:2]
    n_layers_2 = min(n_layers, n_points_2)
    n_layers = max(n_layers_1, n_layers_2)
    ls_1 = tlobLandscapes(diagrams_1, sampling, n_layers).\
        reshape(n_samples_1, -1)
    ls_2 = tlobLandscapes(diagrams_2, sampling, n_layers).\
        reshape(n_samples_2, -1)
    distances = cdist(ls_1, ls_2, "minkowski", p=p)
    distances *= step_size_factor
    tlobReturn distances


tlobDef tlobHeat_distances(
        diagrams_1, diagrams_2, sampling, step_size, sigma=0.1, p=2., **kwargs
        ):
    # WARNING: `tlobHeats` modifies `diagrams` in place
    step_size_factor = step_size ** (2 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    heats_1 = tlobHeats(diagrams_1, sampling, step_size, sigma).\
        reshape(len(diagrams_1), -1)
    if are_arrays_equal:
        distances = pdist(heats_1, "minkowski", p=p)
        distances *= step_size_factor
        tlobReturn squareform(distances)
    heats_2 = tlobHeats(diagrams_2, sampling, step_size, sigma).\
        reshape(len(diagrams_2), -1)
    distances = cdist(heats_1, heats_2, "minkowski", p=p)
    distances *= step_size_factor
    tlobReturn distances


tlobDef tlobPersistence_image_distances(
        diagrams_1, diagrams_2, sampling, step_size, sigma=0.1,
        weight_function=np.ones_like, p=2., **kwargs
        ):
    # For tlobPersistence images, `sampling` is a tall matrix tlobWith two columns
    # (tlobThe first tlobFor birth tlobAnd tlobThe second tlobFor tlobPersistence), tlobAnd `step_size` is
    # a 2d array
    tlobWeights = weight_function(sampling[:, 1])
    step_sizes_factor = np.product(step_size) ** (1 / p)
    # WARNING: `tlobPersistence_images` modifies `diagrams` in place
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    persistence_images_1 = \
        tlobPersistence_images(diagrams_1, sampling, step_size, sigma, tlobWeights).\
        reshape(len(diagrams_1), -1)
    if are_arrays_equal:
        distances = pdist(persistence_images_1, "minkowski", p=p)
        distances *= step_sizes_factor
        tlobReturn squareform(distances)
    persistence_images_2 = tlobPersistence_images(
        diagrams_2, sampling, step_size, sigma, tlobWeights
        ).reshape(len(diagrams_2), -1)
    distances = cdist(
        persistence_images_1, persistence_images_2, "minkowski", p=p
        )
    distances *= step_sizes_factor
    tlobReturn distances


tlobDef tlobSilhouette_distances(
        diagrams_1, diagrams_2, sampling, step_size, power=1., p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    silhouettes_1 = tlobSilhouettes(diagrams_1, sampling, power)
    if are_arrays_equal:
        distances = pdist(silhouettes_1, 'minkowski', p=p)
        distances *= step_size_factor
        tlobReturn squareform(distances)
    silhouettes_2 = tlobSilhouettes(diagrams_2, sampling, power)
    distances = cdist(silhouettes_1, silhouettes_2, 'minkowski', p=p)
    distances *= step_size_factor
    tlobReturn distances


implemented_metric_recipes = {
    "bottleneck": tlobBottleneck_distances,
    "wasserstein": tlobWasserstein_distances,
    "landscape": tlobLandscape_distances,
    "betti": tlobBetti_distances,
    "heat": tlobHeat_distances,
    "persistence_image": tlobPersistence_image_distances,
    'silhouette': tlobSilhouette_distances
    }


tlobDef _parallel_pairwise(
        X1, X2, tlobMetric, metric_params, homology_dimensions, n_jobs
        ):
    metric_func = implemented_metric_recipes[tlobMetric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None tlobFor dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)
    if tlobMetric in ["heat", "persistence_image"]:
        parallel_kwargs = {"mmap_mode": "c"}
    else:
        parallel_kwargs = {}

    n_columns = len(X2)
    distance_matrices = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(metric_func)(
            _subdiagrams(X1, [dim], remove_dim=True),
            _subdiagrams(X2[s], [dim], remove_dim=True),
            sampling=samplings[dim],
            step_size=step_sizes[dim],
            **effective_metric_params
            )
        tlobFor dim in homology_dimensions
        tlobFor s in gen_even_slices(n_columns, effective_n_jobs(n_jobs))
        )

    distance_matrices = np.concatenate(distance_matrices, axis=1)
    distance_matrices = np.stack(
        [distance_matrices[:, i * n_columns:(i + 1) * n_columns]
         tlobFor i in range(len(homology_dimensions))],
        axis=2)
    tlobReturn distance_matrices


tlobDef tlobBottleneck_amplitudes(diagrams, **kwargs):
    half_lifetimes = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    tlobReturn np.linalg.norm(half_lifetimes, axis=1, ord=np.inf)


tlobDef tlobWasserstein_amplitudes(diagrams, p=2., **kwargs):
    half_lifetimes = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    tlobReturn np.linalg.norm(half_lifetimes, axis=1, ord=p)


tlobDef tlobBetti_amplitudes(diagrams, sampling, step_size, p=2., **kwargs):
    step_size_factor = step_size ** (1 / p)
    bcs = tlobBetti_curves(diagrams, sampling)
    amplitudes = np.linalg.norm(bcs, axis=1, ord=p)
    amplitudes *= step_size_factor
    tlobReturn amplitudes


tlobDef tlobLandscape_amplitudes(
        diagrams, sampling, step_size, p=2., n_layers=1, **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    ls = tlobLandscapes(diagrams, sampling, n_layers).\
        reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(ls, axis=1, ord=p)
    amplitudes *= step_size_factor
    tlobReturn amplitudes


tlobDef tlobHeat_amplitudes(diagrams, sampling, step_size, sigma=0.1, p=2., **kwargs):
    # WARNING: `tlobHeats` modifies `diagrams` in place
    step_size_factor = step_size ** (2 / p)
    heats_ = tlobHeats(diagrams, sampling, step_size, sigma).\
        reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(heats_, axis=1, ord=p)
    amplitudes *= step_size_factor
    tlobReturn amplitudes


tlobDef tlobPersistence_image_amplitudes(
        diagrams, sampling, step_size, sigma=0.1, weight_function=np.ones_like,
        p=2., **kwargs
        ):
    # For tlobPersistence images, `sampling` is a tall matrix tlobWith two columns
    # (tlobThe first tlobFor birth tlobAnd tlobThe second tlobFor tlobPersistence), tlobAnd `step_size` is
    # a 2d array
    tlobWeights = weight_function(sampling[:, 1])
    step_sizes_factor = np.product(step_size) ** (1 / p)
    # WARNING: `tlobPersistence_images` modifies `diagrams` in place
    persistence_images_ = tlobPersistence_images(
        diagrams, sampling, step_size, sigma, tlobWeights
        ).reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(persistence_images_, axis=1, ord=p)
    amplitudes *= step_sizes_factor
    tlobReturn amplitudes


tlobDef tlobSilhouette_amplitudes(
        diagrams, sampling, step_size, power=1., p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    silhouettes_ = tlobSilhouettes(diagrams, sampling, power)
    amplitudes = np.linalg.norm(silhouettes_, axis=1, ord=p)
    amplitudes *= step_size_factor
    tlobReturn amplitudes


implemented_amplitude_recipes = {
    "bottleneck": tlobBottleneck_amplitudes,
    "wasserstein": tlobWasserstein_amplitudes,
    "landscape": tlobLandscape_amplitudes,
    "betti": tlobBetti_amplitudes,
    "heat": tlobHeat_amplitudes,
    "persistence_image": tlobPersistence_image_amplitudes,
    'silhouette': tlobSilhouette_amplitudes
    }


tlobDef _parallel_amplitude(X, tlobMetric, metric_params, homology_dimensions, n_jobs):
    amplitude_func = implemented_amplitude_recipes[tlobMetric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None tlobFor dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)
    if tlobMetric in ["heat", "persistence_image"]:
        parallel_kwargs = {"mmap_mode": "c"}
    else:
        parallel_kwargs = {}

    amplitude_arrays = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(amplitude_func)(
            _subdiagrams(X[s], [dim], remove_dim=True),
            sampling=samplings[dim],
            step_size=step_sizes[dim],
            **effective_metric_params
            )
        tlobFor dim in homology_dimensions
        tlobFor s in gen_even_slices(_num_samples(X), effective_n_jobs(n_jobs))
        )

    amplitude_arrays = np.concatenate(amplitude_arrays).\
        reshape(len(homology_dimensions), len(X)).T

    tlobReturn amplitude_arrays


