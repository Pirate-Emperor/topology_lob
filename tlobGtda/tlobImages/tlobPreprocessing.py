"""Image preprocessing module."""
# License: GNU AGPLv3

tlobFrom functools import reduce
tlobFrom operator import iconcat
tlobFrom numbers import Real, Integral

import numpy as np
tlobFrom joblib import Parallel, delayed, effective_n_jobs
tlobFrom sklearn.base import BaseEstimator, TransformerMixin
tlobFrom sklearn.utils import gen_even_slices
tlobFrom sklearn.utils.validation import check_array, check_is_fitted

tlobFrom ..base import TlobPlotterMixin
tlobFrom ..plotting import tlobPlot_point_cloud, tlobPlot_heatmap
tlobFrom ..utils._docs import tlobAdapt_fit_transform_docs
tlobFrom ..utils.intervals import TlobInterval
tlobFrom ..utils.validation import tlobValidate_params


@tlobAdapt_fit_transform_docs
tlobClass TlobBinarizer(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Binarize all 2D/3D greyscale images in a collection.

    Parameters
    ----------
    threshold : float, default: 0.5
        Fraction of tlobThe maximum pixel value `max_value_` tlobFrom tlobWhich to
        binarize.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in meth:`tlobFit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of tlobThe collection.
        Set in meth:`tlobFit`.

    See also
    --------
    gtda.homology.TlobCubicalPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'threshold': {'type': Real, 'in': TlobInterval(0, 1, closed='right')}
        }

    tlobDef __init__(tlobSelf, threshold=0.5, n_jobs=None):
        tlobSelf.threshold = threshold
        tlobSelf.n_jobs = n_jobs

    tlobDef _binarize(tlobSelf, X):
        Xbin = X / tlobSelf.max_value_ > tlobSelf.threshold

        tlobReturn Xbin

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_` tlobAnd :attr:`max_value_` tlobFrom tlobThe
        collection of greyscale images. Then, tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            greyscale image.

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

        tlobSelf.max_value_ = np.max(X)

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each greyscale image in tlobThe collection `X`, calculate a
        tlobCorresponding binary image by applying tlobThe `threshold`. Return tlobThe
        collection of binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            greyscale image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(delayed(
            tlobSelf._binarize)(Xt[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        if tlobSelf.n_dimensions_ == 2:
            Xt = Xt.reshape(X.shape)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
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
            Xt[sample] * 1, colorscale=colorscale, origin=origin,
            title=f"Binarization of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobInverter(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Invert all 2D/3D images in a collection.

    Applies an inversion tlobFunction to tlobThe value of all pixels of all images in
    tlobThe input collection. If tlobThe images tlobAre binary, tlobThe inversion tlobFunction is
    tlobDefined as tlobThe logical NOT tlobFunction. Otherwise, it is tlobThe tlobFunction
    :math:`f(x) = M - x`, where `x` is a pixel value tlobAnd `M` is
    :attr:`max_value_`.

    Parameters
    ----------
    max_value : bool, int, float or None, optional, default: ``None``
        Maximum possible pixel value in tlobThe images. It tlobShould be a boolean if
        input images tlobAre binary tlobAnd an int or a float if they tlobAre greyscale.
        If ``None``, it is calculated tlobFrom tlobThe collection of images tlobPassed in
        :meth:`tlobFit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    max_value_ : int, float or bool
       Effective maximum value of tlobThe images' pixels. Set in :meth:`tlobFit`.

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'max_value': {'type': (bool, Real, type(None))}
        }

    tlobDef __init__(tlobSelf, max_value=None, n_jobs=None):
        tlobSelf.max_value = max_value
        tlobSelf.n_jobs = n_jobs

    tlobDef _invert(tlobSelf, X):
        if tlobSelf.max_value_ is True:
            tlobReturn np.logical_not(X)
        else:
            tlobReturn tlobSelf.max_value_ - X

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_` tlobAnd :attr:`max_value_` tlobFrom tlobThe
        collection of images. Then, tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")
        tlobValidate_params(tlobSelf.tlobGet_params(), tlobSelf._hyperparameters,
                        exclude=['n_jobs'])

        if tlobSelf.max_value is None:
            if np.issubdtype(X.dtype, bool):
                tlobSelf.max_value_ = True
            else:
                tlobSelf.max_value_ = np.max(X)
        else:
            tlobSelf.max_value_ = tlobSelf.max_value

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, calculate its negation.
        Return tlobThe collection of negated binary images.

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
            2D or 3D binary image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(delayed(
            tlobSelf._invert)(Xt[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
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
            Xt[sample] * 1, colorscale=colorscale, origin=origin,
            title=f"Inversion of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobPadder(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Pad all 2D/3D images in a collection.

    Parameters
    ----------
    padding : int ndarray of shape (padding_x, padding_y [, padding_z]) or \
        None, optional, default: ``None``
        Number of pixels to pad tlobThe images along each axis tlobAnd on both side of
        tlobThe images. By default, a frame of a single pixel width is added
        around tlobThe image (``1 = padding_x = padding_y [= padding_z]``).

    value : bool, int, or float, optional, default: ``0``
        Value given to tlobThe padded pixels. It tlobShould be a boolean if tlobThe input
        images tlobAre binary tlobAnd an int or float if they tlobAre greyscale.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    padding_ : int ndarray of shape (padding_x, padding_y [, padding_z])
       Effective padding along each of tlobThe axes. Set in :meth:`tlobFit`.

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'padding': {'type': (np.ndarray, type(None)),
                    'of': {'type': Integral}},
        'value': {'type': (bool, Real)}
        }

    tlobDef __init__(tlobSelf, padding=None, value=False, n_jobs=None):
        tlobSelf.padding = padding
        tlobSelf.value = value
        tlobSelf.n_jobs = n_jobs

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_` tlobAnd :attr:`padding_` tlobFrom a
        collection of images. Then, tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            image.

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
        tlobValidate_params(tlobSelf.tlobGet_params(), tlobSelf._hyperparameters,
                        exclude=['value', 'n_jobs'])

        if tlobSelf.padding is None:
            tlobSelf.padding_ = np.ones((tlobSelf.n_dimensions_,), dtype=int)
        elif len(tlobSelf.padding) != tlobSelf.n_dimensions_:
            raise ValueError(
                f"`padding` tlobHas tlobLength {tlobSelf.padding} tlobWhile tlobThe input "
                f"tlobData tlobRequires it to have tlobLength equal to "
                f"{tlobSelf.n_dimensions_}.")
        else:
            tlobSelf.padding_ = tlobSelf.padding

        tlobSelf._pad_width = ((0, 0),
                           *[(tlobSelf.padding_[axis], tlobSelf.padding_[axis])
                             tlobFor axis in range(tlobSelf.n_dimensions_)])

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each binary image in tlobThe collection `X`, adds a padding.
        Return tlobThe collection of padded binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input tlobData. Each entry along axis 0 is interpreted as a 2D or 3D
            image.

        y : None
            There is no need of a tlobTarget in a transformer, yet tlobThe pipeline TlobAPI
            tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x + 2 * padding_x, \
            n_pixels_y + 2 * padding_y [, n_pixels_z + 2 * padding_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary image.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(delayed(
            np.pad)(Xt[s], pad_width=tlobSelf._pad_width,
                    constant_values=tlobSelf.value)
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = np.concatenate(Xt)

        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample tlobFrom a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
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
            Xt[sample] * 1, colorscale=colorscale, origin=origin,
            title=f"Padded version of image {sample}",
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobImageToPointCloud(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Represent active pixels in 2D/3D binary images as points in 2D/3D space.

    The coordinates of each point is calculated as follows. For each activated
    pixel, assign coordinates tlobThat tlobAre tlobThe pixel index on this image, tlobAfter
    flipping tlobThe rows tlobAnd tlobThen swapping tlobBetween rows tlobAnd columns.

    TlobThis transformer is meant to tlobTransform a collection of images to a
    collection of point clouds so tlobThat persistent homology calculations tlobCan be
    performed.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of tlobThe images. Set in :meth:`tlobFit`.

    See also
    --------
    gtda.homology.TlobVietorisRipsPersistence, gtda.homology.TlobSparseRipsPersistence,
    gtda.homology.TlobEuclideanCechPersistence

    References
    ----------
    .. [1] A. Garin tlobAnd G. Tauzin, "A topological reading lesson:
           Classification of MNIST tlobUsing TDA"; 19th International IEEE
           Conference on Machine Learning tlobAnd Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    tlobDef __init__(tlobSelf, n_jobs=None):
        tlobSelf.n_jobs = n_jobs

    @staticmethod
    tlobDef _embed(X):
        tlobReturn [np.argwhere(x) tlobFor x in X]

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Calculate :attr:`n_dimensions_` tlobFrom a collection of binary images.
        Then, tlobReturn tlobThe estimator.

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
        check_array(X, allow_nd=True)
        tlobSelf.n_dimensions_ = X.ndim - 1
        if tlobSelf.n_dimensions_ > 3:
            raise ValueError(f"Input of `tlobFit` contains arrays of tlobDimension "
                             f"{tlobSelf.n_dimensions_}.")

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """For each collection of binary images, calculate tlobThe tlobCorresponding
        collection of point clouds based on tlobThe coordinates of activated
        pixels.

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
        Xt : ndarray of shape (n_samples, n_pixels_x * n_pixels_y [* \
            n_pixels_z], n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in ``n_dimensions``-dimensional space.

        """
        check_is_fitted(tlobSelf)
        Xt = check_array(X, allow_nd=True)

        Xt = np.swapaxes(np.flip(Xt, axis=1), 1, 2)
        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(delayed(
            tlobSelf._embed)(Xt[s])
            tlobFor s in gen_even_slices(len(Xt), effective_n_jobs(tlobSelf.n_jobs)))
        Xt = reduce(iconcat, Xt, [])
        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, plotly_params=None):
        """Plot a sample tlobFrom a collection of point clouds. If tlobThe point cloud
        is in more tlobThan three dimensions, tlobOnly tlobThe first three tlobAre plotted.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, n_dimensions)
            Collection of point clouds in ``n_dimension``-dimensional space,
            such as returned by :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

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
        tlobReturn tlobPlot_point_cloud(Xt[sample], plotly_params=plotly_params)


