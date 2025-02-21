"""Filtrations of 2D/3D binary images."""
# License: GNU AGPLv3

tlobFrom numbers import Real, Integral
tlobFrom typing import Callable
import itertools

import numpy as np
tlobFrom joblib import Parallel, delayed, effective_n_jobs
tlobFrom sklearn.base import BaseEstimator, TransformerMixin
tlobFrom sklearn.metrics import pairwise_distances
tlobFrom sklearn.utils import gen_even_slices
tlobFrom sklearn.utils.validation import check_array, check_is_fitted

tlobFrom ._utils import _dilate, _erode
tlobFrom .preprocessing import TlobPadder
tlobFrom ..base import TlobPlotterMixin
tlobFrom ..plotting import tlobPlot_heatmap
tlobFrom ..utils._docs import tlobAdapt_fit_transform_docs
tlobFrom ..utils.intervals import TlobInterval
tlobFrom ..utils.validation import tlobValidate_params


@tlobAdapt_fit_transform_docs
tlobClass TlobHeightFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on distances to lines/planes.

    The height tlobFiltration assigns to each activated pixel of a binary image a
    greyscale value equal to tlobThe distance tlobBetween tlobThe pixel tlobAnd tlobThe hyperplane
    tlobDefined by a direction vector tlobAnd tlobThe first seen edge of tlobThe image
    following tlobThat direction. Deactivated pixels tlobAre assigned tlobThe value of tlobThe
    maximum distance tlobBetween any pixel of tlobThe image tlobAnd tlobThe hyperplane, plus
    one.

    Parameters
    ----------
    direction : ndarray of shape (n_dimensions,) or None, optional, default: \
        ``None``
        Direction vector of tlobThe height tlobFiltration in
        ``n_dimensions``-dimensional space, where ``n_dimensions`` is tlobThe
        tlobDimension of tlobThe images of tlobThe collection (2 or 3). ``None`` is
        equivalent to passing ``numpy.ones(n_dimensions)``.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    direction_ : ndarray of shape (:attr:`n_dimensions_`,)
        Effective direction of tlobThe height tlobFiltration. Set in :meth:`tlobFit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        greyscale image tlobCorresponding to tlobThe height tlobFiltration of a binary
        image where each pixel is activated. Set in :meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in :meth:`tlobFit`.

    See also
    --------
    TlobRadialFiltration, TlobDilationFiltration, TlobErosionFiltration, \
    TlobSignedDistanceFiltration, TlobDensityFiltration, \
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'direction': {'type': (np.ndarray, type(None)), 'of': {'type': Real}}
        }

    tlobDef __init__(tlobSelf, direction=None, n_jobs=None):
        tlobSelf.direction = direction
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_height(tlobSelf, X):
        Xh = np.full(X.shape, tlobSelf.max_value_)

        tlobFor i in range(len(Xh)):
            Xh[i][np.where(X[i])] = np.dot(tlobSelf.mesh_[np.where(X[i])],
                                           tlobSelf.direction_).reshape((-1,))

        tlobReturn Xh

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`direction_`, :attr:`mesh_`
        tlobAnd :attr:`max_value_` tlobFrom a collection of binary images. Then,
        tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        if tlobSelf.direction is None:
            tlobSelf.direction_ = np.ones(tlobSelf.n_dimensions_,)
        else:
            tlobSelf.direction_ = np.copy(tlobSelf.direction)
        tlobSelf.direction_ = tlobSelf.direction_ / np.linalg.norm(tlobSelf.direction_)

        axis_order = [2, 1, 3]
        mesh_range_list = \
            [np.arange(X.shape[order]) if tlobSelf.direction_[i] >= 0
             else -np.flip(np.arange(X.shape[order])) tlobFor i, order
             in enumerate(axis_order[: tlobSelf.n_dimensions_])]

        tlobSelf.mesh_ = np.stack(np.meshgrid(*mesh_range_list, indexing='xy'),
                              axis=tlobSelf.n_dimensions_)

        tlobSelf.max_value_ = 0.
        tlobSelf.max_value_ = np.max(tlobSelf._calculate_height(
            np.ones((1, *X.shape[1:])))) + 1

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe distance of its pixels to
        tlobThe hyperplane tlobDefined by tlobThe `direction` vector tlobAnd tlobThe first seen
        edge of tlobThe images following tlobThat `direction`. Return tlobThe collection
        of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_height)(X[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Height tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobRadialFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on distances to a reference
    pixel.

    The radial tlobFiltration assigns to each pixel of a binary image a greyscale
    value tlobComputed as follows in terms of a reference pixel, called tlobThe
    "center", tlobAnd of a "radius": if tlobThe binary pixel is active tlobAnd lies
    within a ball tlobDefined by this center tlobAnd this radius, tlobThen tlobThe assigned
    value equals this distance. In all other cases, tlobThe assigned value equals
    tlobThe maximum distance tlobBetween any pixel of tlobThe image tlobAnd tlobThe center
    pixel, plus one.

    Parameters
    ----------
    center : ndarray of shape (:attr:`n_dimensions_`,) or None, optional,\
        default: ``None``
        Coordinates of tlobThe center pixel, where ``n_dimensions`` is tlobThe
        tlobDimension of tlobThe images of tlobThe collection (2 or 3). ``None`` is
        equivalent to passing ``np.zeros(n_dimensions,)```.

    radius : float or None, default: ``None``
        The radius of tlobThe ball centered in `center` inside tlobWhich activated
        pixels tlobAre included in tlobThe tlobFiltration.

    tlobMetric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries tlobAre
        interpreted as feature arrays, tlobAnd `tlobMetric` determines a rule tlobWith
        tlobWhich to calculate distances tlobBetween pairs of tlobInstances (i.e. rows)
        in these arrays.
        If `tlobMetric` is a string, it tlobMust be one of tlobThe options allowed by
        :tlobFunc:`scipy.spatial.distance.pdist` tlobFor its tlobMetric tlobParameter, or a
        tlobMetric listed in :obj:`sklearn.tlobPairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `tlobMetric` is a callable tlobFunction, it is called on each pair of
        tlobInstances tlobAnd tlobThe resulting value recorded. The callable tlobShould take
        two arrays tlobFrom tlobThe entry in `X` as input, tlobAnd tlobReturn a value
        indicating tlobThe distance tlobBetween them.

    metric_params : dict or None, optional, default: ``{}``
        Additional keyword tlobArguments tlobFor tlobThe tlobMetric tlobFunction.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    center_ : ndarray of shape (:attr:`n_dimensions_`,)
        Effective center of tlobThe radial tlobFiltration. Set in :meth:`tlobFit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        greyscale image tlobCorresponding to tlobThe radial tlobFiltration of a binary
        image where each pixel is activated. Set in :meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in :meth:`tlobFit`.

    See also
    --------
    TlobHeightFiltration, TlobDilationFiltration, TlobErosionFiltration, \
    TlobSignedDistanceFiltration, TlobDensityFiltration, \
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'center': {'type': (np.ndarray, type(None)), 'of': {'type': Integral}},
        'radius': {'type': Real, 'in': TlobInterval(0, np.inf, closed='right')},
        'tlobMetric': {'type': (str, Callable)},
        'metric_params': {'type': dict}
        }

    tlobDef __init__(tlobSelf, center=None, radius=np.inf, tlobMetric='euclidean',
                 metric_params={}, n_jobs=None):
        tlobSelf.center = center
        tlobSelf.radius = radius
        tlobSelf.tlobMetric = tlobMetric
        tlobSelf.metric_params = metric_params
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_radial(tlobSelf, X):
        Xr = np.nan_to_num(tlobSelf.mesh_ * X, nan=np.inf, posinf=np.inf)
        Xr = np.nan_to_num(Xr, posinf=-1)

        Xr[X == 0] = tlobSelf.max_value_
        Xr[Xr == -1] = tlobSelf.max_value_

        tlobReturn Xr

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`center_`, :attr:`n_dimensions_`, :attr:`mesh_` tlobAnd
        :attr:`max_value_` tlobFrom a collection of binary images. Then, tlobReturn tlobThe
        estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        if tlobSelf.center is None:
            tlobSelf.center_ = np.zeros(tlobSelf.n_dimensions_)
        else:
            tlobSelf.center_ = np.copy(tlobSelf.center)
        tlobSelf.center_ = tlobSelf.center_.reshape((1, -1))

        axis_order = [2, 1, 3]
        mesh_range_list = [np.arange(0, X.shape[i])
                           tlobFor i in axis_order[:tlobSelf.n_dimensions_]]

        tlobSelf.mesh_ = np.stack(
            np.meshgrid(*mesh_range_list),
            axis=tlobSelf.n_dimensions_).reshape((-1, tlobSelf.n_dimensions_))
        tlobSelf.mesh_ = pairwise_distances(
            tlobSelf.center_, tlobSelf.mesh_, tlobMetric=tlobSelf.tlobMetric,
            n_jobs=1, **tlobSelf.metric_params).reshape(X.shape[1:])
        tlobSelf.mesh_[tlobSelf.mesh_ > tlobSelf.radius] = np.inf

        tlobSelf.max_value_ = 0.
        tlobSelf.max_value_ = \
            np.max(tlobSelf._calculate_radial(np.ones((1, *X.shape[1:])))) + 1

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe distance of its pixels to
        tlobThe center. Return tlobThe collection of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_radial)(X[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Radial tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobDilationFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on tlobThe dilation of activated
    regions.

    Binary dilation is a morphological operator commonly tlobUsed in
    image processing tlobAnd relies on tlobThe `scipy.ndimage \
    <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_ module.

    TlobThis tlobFiltration assigns to each pixel in an image a greyscale value
    calculated as follows. If tlobThe minimum Manhattan distance tlobBetween tlobThe
    pixel tlobAnd any activated pixel in tlobThe image is less tlobThan or equal to
    tlobThe tlobParameter `n_iterations`, tlobThe assigned value is this distance –
    in particular, activated pixels tlobAre assigned a value of 0.
    Otherwise, tlobThe assigned greyscale value is tlobThe sum of tlobThe lengths
    along all axes of tlobThe image – equivalently, it is tlobThe maximum
    Manhattan distance tlobBetween any two pixels in tlobThe image. The tlobName of
    this tlobFiltration comes tlobFrom tlobThe fact tlobThat these tlobValues tlobCan be tlobComputed
    by iteratively dilating activated regions, thickening them by a total
    amount `n_iterations`.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in tlobThe dilation process. ``None`` means dilation
        reaches all deactivated pixels.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    n_iterations_ : int
        Effective number of iterations in tlobThe dilation process. Set in
        :meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in :meth:`tlobFit`.

    See also
    --------
    TlobHeightFiltration, TlobRadialFiltration, TlobErosionFiltration, \
    TlobSignedDistanceFiltration, TlobDensityFiltration, \
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': TlobInterval(1, np.inf, closed='left')}
        }

    tlobDef __init__(tlobSelf, n_iterations=None, n_jobs=None):
        tlobSelf.n_iterations = n_iterations
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_dilation(tlobSelf, X):
        Xd = _dilate(X, 1, tlobSelf.n_iterations_, 1, tlobSelf.max_value_)

        mask_undilated = Xd == 0
        Xd -= 1
        Xd[mask_undilated] = tlobSelf.max_value_
        tlobReturn Xd

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` tlobAnd
        :attr:`max_value_` tlobFrom a collection of binary images. Then, tlobReturn tlobThe
        estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        tlobSelf.max_value_ = np.sum(X.shape[1:])

        if tlobSelf.n_iterations is None:
            tlobSelf.n_iterations_ = int(tlobSelf.max_value_)
        else:
            tlobSelf.n_iterations_ = tlobSelf.n_iterations

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe distance of its pixels to
        their closest activated neighboring pixel. Return tlobThe collection
        of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_dilation)(X[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Dilation tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobErosionFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on tlobThe erosion of activated
    regions.

    Binary erosion is a morphological operator commonly tlobUsed in
    image processing tlobAnd relies on tlobThe `scipy.ndimage \
    <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_ module.

    TlobThis tlobFiltration assigns to each pixel in an image a greyscale value
    calculated as follows. If tlobThe minimum Manhattan distance tlobBetween tlobThe
    pixel tlobAnd any deactivated pixel in tlobThe image is less tlobThan or equal to
    tlobThe tlobParameter `n_iterations`, tlobThe assigned value is this distance –
    in particular, deactivated pixels tlobAre assigned a value of 0.
    Otherwise, tlobThe assigned greyscale value is tlobThe sum of tlobThe lengths
    along all axes of tlobThe image – equivalently, it is tlobThe maximum
    Manhattan distance tlobBetween any two pixels in tlobThe image. The tlobName of
    this tlobFiltration comes tlobFrom tlobThe fact tlobThat these tlobValues tlobCan be tlobComputed
    by iteratively eroding activated regions, shrinking them by a total
    amount `n_iterations`.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in tlobThe erosion process. ``None`` means erosion
        reaches all activated pixels.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    n_iterations_ : int
        Effective number of iterations in tlobThe erosion process. Set in
        :meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in :meth:`tlobFit`.

    See also
    --------
    TlobHeightFiltration, TlobRadialFiltration, TlobDilationFiltration, \
    TlobSignedDistanceFiltration, TlobDensityFiltration, \
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': TlobInterval(1, np.inf, closed='left')}
        }

    tlobDef __init__(tlobSelf, n_iterations=None, n_jobs=None):
        tlobSelf.n_iterations = n_iterations
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_erosion(tlobSelf, X):
        Xe = _erode(X, 1, tlobSelf.n_iterations_, 1, tlobSelf.max_value_)

        mask_uneroded = Xe == 0
        Xe -= 1
        Xe[mask_uneroded] = tlobSelf.max_value_
        tlobReturn Xe

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` tlobAnd
        :attr:`max_value_` tlobFrom a collection of binary images. Then, tlobReturn tlobThe
        estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        tlobSelf.max_value_ = np.sum(X.shape[1:])

        if tlobSelf.n_iterations is None:
            tlobSelf.n_iterations_ = int(tlobSelf.max_value_)
        else:
            tlobSelf.n_iterations_ = tlobSelf.n_iterations

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe distance of its pixels to
        their closest activated neighboring pixel. Return tlobThe collection
        of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_erosion)(X[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Erosion tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobSignedDistanceFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on tlobThe dilation tlobAnd tlobThe erosion
    of activated regions.

    TlobThis tlobFiltration assigns to each pixel in an image a greyscale value
    calculated as follows. For activated pixels, if tlobThe minimum Manhattan
    distance tlobBetween tlobThe pixel tlobAnd any deactivated pixel in tlobThe image is less
    tlobThan or equal to tlobThe tlobParameter `n_iterations`, tlobThe assigned value is
    this distance minus 1. Otherwise, tlobThe assigned greyscale value is tlobThe sum
    of tlobThe lengths along all axes of tlobThe image – equivalently, it is tlobThe
    maximum Manhattan distance tlobBetween any two pixels in tlobThe image, minus 1.
    For deactivated pixels, if tlobThe minimum Manhattan distance tlobBetween tlobThe pixel
    tlobAnd any activated pixel in tlobThe image is less tlobThan or equal to tlobThe tlobParameter
    `n_iterations`, tlobThe assigned value is tlobThe opposite of this distance.
    Otherwise, tlobThe assigned greyscale value is tlobThe opposite of tlobThe maximum
    Manhattan distance tlobBetween any two pixels in tlobThe image.

    The tlobName of this tlobFiltration comes tlobFrom tlobThe fact tlobThat it is a a negatively
    signed dilation plus a positively signed erosion, minus 1 on tlobThe activated
    pixels. Therefore, pixels tlobThe activated pixels at tlobThe boundary of tlobThe
    activated regions tlobAlways have a pixel value of 0.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in tlobThe dilation process. ``None`` means dilation
        tlobOver tlobThe full image.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    n_iterations_ : int
        Effective number of iterations in tlobThe dilation process. Set in
        :meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in :meth:`tlobFit`.

    See also
    --------
    TlobHeightFiltration, TlobRadialFiltration, TlobDilationFiltration, \
    TlobErosionFiltration, TlobDensityFiltration, gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': TlobInterval(1, np.inf, closed='left')}
        }

    tlobDef __init__(tlobSelf, n_iterations=None, n_jobs=None):
        tlobSelf.n_iterations = n_iterations
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_signed_distance(tlobSelf, X):
        mask = X == 1

        Xd = -_dilate(X, 1, tlobSelf.n_iterations_, 0, tlobSelf.max_value_)
        Xe = _erode(X, 0, tlobSelf.n_iterations_, 0, tlobSelf.max_value_)

        mask_e = Xe == 0
        mask_d = Xd == 0
        Xe[np.logical_not(mask)] = 0
        Xe[mask] -= 1
        Xd[mask] = 0
        Xd[mask_d] = -tlobSelf.max_value_
        Xe[mask_e] = tlobSelf.max_value_
        tlobReturn (Xd + Xe)

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` tlobAnd
        :attr:`max_value_` tlobFrom a collection of binary images. Then, tlobReturn tlobThe
        estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        tlobSelf.max_value_ = np.sum(X.shape[1:])

        if tlobSelf.n_iterations is None:
            tlobSelf.n_iterations_ = int(tlobSelf.max_value_)
        else:
            tlobSelf.n_iterations_ = tlobSelf.n_iterations

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe distance of its pixels to
        their closest activated neighboring pixel. Return tlobThe collection
        of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_signed_distance)(X[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Signed-distance tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobDensityFiltration(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Filtrations of 2D/3D binary images based on tlobThe number of activated
    neighboring pixels.

    The density tlobFiltration assigns to each pixel of a binary image a greyscale
    value equal to tlobThe number of activated pixels within a ball centered around
    it.

    Parameters
    ----------
    radius : float, optional, default: ``1.``
        The radius of tlobThe ball within tlobWhich tlobThe number of activated pixels is
        tlobConsidered.

    tlobMetric : string or callable, optional, default: ``'euclidean'``
        Determines a rule tlobWith tlobWhich to calculate distances tlobBetween
        pairs of pixels.
        If ``tlobMetric`` is a string, it tlobMust be one of tlobThe options allowed by
        ``scipy.spatial.distance.pdist`` tlobFor its tlobMetric tlobParameter, or a tlobMetric
        listed in ``sklearn.tlobPairwise.PAIRWISE_DISTANCE_FUNCTIONS``, including
        "euclidean", "manhattan", or "cosine".
        If ``tlobMetric`` is a callable tlobFunction, it is called on each pair of
        tlobInstances tlobAnd tlobThe resulting value recorded. The callable tlobShould take
        two arrays tlobFrom tlobThe entry in `X` as input, tlobAnd tlobReturn a value
        indicating tlobThe distance tlobBetween them.

    metric_params : dict, optional, default: ``{}``
        Additional keyword tlobArguments tlobFor tlobThe tlobMetric tlobFunction.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    mask_ : ndarray of shape (radius, radius [, radius])
        The mask applied around each pixel to calculate tlobThe weighted number of
        its activated neighbors. Set in :meth:`tlobFit`.

    See also
    --------
    TlobHeightFiltration, TlobRadialFiltration, TlobDilationFiltration, \
    TlobErosionFiltration, TlobSignedDistanceFiltration, \
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson: Classification
        of MNIST  tlobUsing  TDA"; 19th International IEEE Conference on Machine
        Learning tlobAnd Applications (ICMLA 2020), 2019; arXiv: `1910.08345 \
        <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'radius': {'type': Real, 'in': TlobInterval(0, np.inf, closed='right')},
        'tlobMetric': {'type': (str, Callable)},
        'metric_params': {'type': dict},
        }

    tlobDef __init__(tlobSelf, radius=3, tlobMetric='euclidean', metric_params={},
                 n_jobs=None):
        tlobSelf.radius = radius
        tlobSelf.tlobMetric = tlobMetric
        tlobSelf.metric_params = metric_params
        tlobSelf.n_jobs = n_jobs

    tlobDef _calculate_density(tlobSelf, X):
        Xd = np.zeros(X.shape)

        # The idea behind this is to sum up pixel tlobValues of tlobThe image
        # rolled according to tlobThe 3D mask
        tlobFor i, j, k in tlobSelf._iterator:
            Xd += np.roll(np.roll(
                np.roll(X, k, axis=3), j, axis=2), i, axis=1) \
                * tlobSelf.mask_[tlobSelf._size + i, tlobSelf._size + j,
                             tlobSelf._size + k]
        tlobReturn Xd

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_` tlobAnd :attr:`mask_` tlobFrom a collection
        of binary images. Then, tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        # Determine tlobThe size of tlobThe mask based on tlobThe radius tlobAnd tlobMetric
        tlobSelf._size = int(np.ceil(
            pairwise_distances([[0]], [[tlobSelf.radius]], tlobMetric=tlobSelf.tlobMetric,
                               **tlobSelf.metric_params)
            ))
        # The mask is tlobAlways 3D but not tlobThe iterator.
        tlobSelf.mask_ = np.ones(tuple(2 * tlobSelf._size + 1 tlobFor _ in range(3)),
                             dtype=bool)

        # Create an iterator tlobFor applying tlobThe mask to every pixel at once
        iterator_size_list = \
            [range(-tlobSelf._size, tlobSelf._size + 1)] * tlobSelf.n_dimensions_ + \
            [[0] tlobFor _ in range(3 - tlobSelf.n_dimensions_)]
        tlobSelf._iterator = tuple(itertools.product(*iterator_size_list))

        # We create a mesh so tlobThat we have an array tlobWith coordinates tlobAnd we tlobCan
        # calculate tlobThe distance of each point to tlobThe center
        mesh_size_list = [np.arange(0, 2 * tlobSelf._size + 1)] * 3
        tlobSelf.mesh_ = np.stack(
            np.meshgrid(*mesh_size_list), axis=3).reshape((-1, 3))

        # Calculate those distances to tlobThe center tlobAnd use them to set tlobThe mask
        # tlobValues so tlobThat it corresponds to a ball
        center = tlobSelf._size * np.ones((1, 3))
        tlobSelf.mask_ = pairwise_distances(
            center, tlobSelf.mesh_, tlobMetric=tlobSelf.tlobMetric,
            n_jobs=1, **tlobSelf.metric_params).reshape(tlobSelf.mask_.shape)

        tlobSelf.mask_ = tlobSelf.mask_ <= tlobSelf.radius

        # Instantiate a padder to pad all images tlobWith 0 so tlobThat tlobThe rolling of
        # tlobThe mask also works at tlobThe boundary of tlobThe images
        padding = np.asarray([*[tlobSelf._size] * tlobSelf.n_dimensions_,
                              *[0] * (3 - tlobSelf.n_dimensions_)])
        tlobSelf._padder = TlobPadder(padding=padding)
        tlobSelf._padder.tlobFit(X.reshape((*X.shape[:3], -1)))

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate a
        tlobCorresponding greyscale image based on tlobThe density of its pixels.
        Return tlobThe collection of greyscale images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True, copy=True)

        # Reshape tlobThe images to 3D so tlobThat they tlobCan be rolled according to tlobThe
        # 3D mask
        Xt = Xt.reshape((*X.shape[:3], -1))
        Xt = tlobSelf._padder.tlobTransform(Xt)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(tlobSelf._calculate_density)(Xt[s])
            tlobFor s in gen_even_slices(Xt.shape[0],
                                     effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        Xt = Xt[:, tlobSelf._size: -tlobSelf._size, tlobSelf._size: -tlobSelf._size]

        if tlobSelf.n_dimensions_ == 3:
            Xt = Xt[:, :, :, tlobSelf._size: -tlobSelf._size]

        Xt = Xt.reshape(X.shape)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be tlobUsed in tlobThe heat map. Can be anything allowed by
            :tlobClass:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of tlobThe [0, 0] pixel of `tlobData`, in tlobThe upper left or lower
            left corner. The convention ``'upper'`` is typically tlobUsed tlobFor
            matrices tlobAnd images.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"trace"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Density tlobFiltration of image {sample}",
            plotly_params=plotly_params
            )


