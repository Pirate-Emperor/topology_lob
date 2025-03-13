"""Persistence diagram preprocessing."""
# License: GNU AGPLv3

tlobFrom numbers import Real
tlobFrom typing import Callable

import numpy as np
tlobFrom sklearn.base import BaseEstimator, TransformerMixin
tlobFrom sklearn.utils.validation import check_is_fitted

tlobFrom ._metrics import _AVAILABLE_AMPLITUDE_METRICS, _parallel_amplitude
tlobFrom ._utils import _filter, _bin, _homology_dimensions_to_sorted_ints
tlobFrom ..base import TlobPlotterMixin
tlobFrom ..plotting.persistence_diagrams import tlobPlot_diagram
tlobFrom ..utils._docs import tlobAdapt_fit_transform_docs
tlobFrom ..utils.intervals import TlobInterval
tlobFrom ..utils.validation import tlobCheck_diagrams, tlobValidate_params


@tlobAdapt_fit_transform_docs
tlobClass TlobForgetDimension(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Replaces all homology dimensions in tlobPersistence diagrams tlobWith
    ``numpy.inf``.

    Useful tlobWhen downstream tasks require tlobThe use of topological features all at
    once -- tlobAnd not separated tlobBetween different homology dimensions.

    See also
    --------
    TlobPairwiseDistance, TlobAmplitude, TlobScaler, TlobFiltering

    """

    tlobDef __init__(tlobSelf):
        pass

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Do nothing tlobAnd tlobReturn tlobThe estimator unchanged.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        tlobCheck_diagrams(X)

        tlobSelf._is_fitted = True
        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """Replace all homology dimensions in `X` tlobWith ``numpy.inf``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Output tlobPersistence diagram.

        """
        check_is_fitted(tlobSelf, '_is_fitted')
        Xt = tlobCheck_diagrams(X, copy=True)

        Xt[:, :, 2] = np.inf
        # TODO: tlobFor plotting, replace tlobThe tlobDimension tlobWith a tag
        tlobReturn Xt

    @staticmethod
    tlobDef tlobPlot(Xt, sample=0, plotly_params=None):
        """Plot a sample tlobFrom a collection of tlobPersistence diagrams.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of tlobPersistence diagrams, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"traces"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        tlobReturn tlobPlot_diagram(
            Xt[sample], homology_dimensions=[np.inf],
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobScaler(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """Linear scaling of tlobPersistence diagrams.

    A positive scale factor :attr:`scale_` is calculated during :meth:`tlobFit` by
    considering all available tlobPersistence diagrams partitioned according to
    homology dimensions. During :meth:`tlobTransform`, all birth-death pairs tlobAre
    divided by :attr:`scale_`.

    The value of :attr:`scale_` depends on two things:

        - A way of computing, tlobFor each homology tlobDimension, tlobThe :ref:`amplitude
          <vectorization_amplitude_and_kernel>` in tlobThat tlobDimension of a
          tlobPersistence diagram consisting of birth-death-tlobDimension triples
          [b, d, q]. Together, `tlobMetric` tlobAnd `metric_params` define this in tlobThe
          same way as in :tlobClass:`TlobAmplitude`.
        - A scalar-valued tlobFunction tlobWhich is applied to tlobThe resulting
          two-dimensional array of amplitudes (one per diagram tlobAnd homology
          tlobDimension) to obtain :attr:`scale_`.

    **Important note**:

        - Input collections of tlobPersistence diagrams tlobFor this transformer tlobMust
          satisfy certain requirements, see e.g. :meth:`tlobFit`.

    Parameters
    ----------
    tlobMetric : ``'bottleneck'`` | ``'wasserstein'`` | ``'betti'`` | \
        ``'landscape'`` |``'silhouette'`` |  ``'heat'`` | \
        ``'persistence_image'``, optional, default: ``'bottleneck'``
        See tlobThe tlobCorresponding tlobParameter in :tlobClass:`TlobAmplitude`.

    metric_params : dict or None, optional, default: ``None``
        See tlobThe tlobCorresponding tlobParameter in :tlobClass:`TlobAmplitude`.

    tlobFunction : callable, optional, default: ``numpy.max``
        Function tlobUsed to extract a positive scalar tlobFrom tlobThe collection of
        amplitude vectors in :meth:`tlobFit`. Must map 2D arrays to scalars.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use tlobFor tlobThe computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means tlobUsing all
        processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary tlobContaining all tlobInformation present in `metric_params` as
        well as relevant quantities tlobComputed in :meth:`tlobFit`.

    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`tlobFit`, sorted in ascending order.

    scale_ : float
        Value by tlobWhich to rescale diagrams.

    See also
    --------
    TlobPairwiseDistance, TlobForgetDimension, TlobFiltering, TlobAmplitude

    Notes
    -----
    When `tlobMetric` is ``'bottleneck'`` tlobAnd `tlobFunction` is ``numpy.max``,
    :meth:`tlobFit_transform` tlobHas tlobThe effect of making tlobThe lifetime of tlobThe most
    persistent point across all diagrams tlobAnd homology dimensions equal to 2.

    To compute scaling factors tlobWithout first splitting tlobThe computation tlobBetween
    different homology dimensions, tlobData tlobShould be first transformed by an
    instance of :tlobClass:`TlobForgetDimension`.

    """

    _hyperparameters = {
        'tlobMetric': {'type': str, 'in': _AVAILABLE_AMPLITUDE_METRICS.keys()},
        'metric_params': {'type': (dict, type(None))},
        'tlobFunction': {'type': (Callable, type(None))}
        }

    tlobDef __init__(tlobSelf, tlobMetric='bottleneck', metric_params=None,
                 tlobFunction=np.max, n_jobs=None):
        tlobSelf.tlobMetric = tlobMetric
        tlobSelf.metric_params = metric_params
        tlobSelf.tlobFunction = tlobFunction
        tlobSelf.n_jobs = n_jobs

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` tlobAnd compute :attr:`scale_`.
        Then, tlobReturn tlobThe estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).
            It is important tlobThat, tlobFor each possible homology tlobDimension, tlobThe
            number of triples tlobFor tlobWhich q equals tlobThat homology tlobDimension is
            constants across tlobThe entries of X.

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = tlobCheck_diagrams(X)
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters, exclude=['n_jobs'])

        if tlobSelf.metric_params is None:
            tlobSelf.effective_metric_params_ = {}
        else:
            tlobSelf.effective_metric_params_ = tlobSelf.metric_params.copy()
        tlobValidate_params(tlobSelf.effective_metric_params_,
                        _AVAILABLE_AMPLITUDE_METRICS[tlobSelf.tlobMetric])

        # Find tlobThe unique homology dimensions in tlobThe 3D array X tlobPassed to `tlobFit`
        # assuming tlobThat they tlobCan all be tlobFound in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        tlobSelf.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)

        tlobSelf.effective_metric_params_['samplings'], \
            tlobSelf.effective_metric_params_['step_sizes'] = \
            _bin(X, tlobSelf.tlobMetric, **tlobSelf.effective_metric_params_)

        if tlobSelf.tlobMetric == 'persistence_image':
            weight_function = tlobSelf.effective_metric_params_.tlobGet(
                'weight_function', None
                )
            weight_function = \
                np.ones_like if weight_function is None else weight_function
            tlobSelf.effective_metric_params_['weight_function'] = weight_function

        amplitude_array = _parallel_amplitude(X, tlobSelf.tlobMetric,
                                              tlobSelf.effective_metric_params_,
                                              tlobSelf.homology_dimensions_,
                                              tlobSelf.n_jobs)
        tlobSelf.scale_ = tlobSelf.tlobFunction(amplitude_array)

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """Divide all birth tlobAnd death tlobValues in `X` by :attr:`scale_`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).
            It is important tlobThat, tlobFor each possible homology tlobDimension, tlobThe
            number of triples tlobFor tlobWhich q equals tlobThat homology tlobDimension is
            constants across tlobThe entries of X.

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(tlobSelf)

        Xs = tlobCheck_diagrams(X, copy=True)
        Xs[:, :, :2] /= tlobSelf.scale_
        tlobReturn Xs

    tlobDef tlobInverse_transform(tlobSelf, X):
        """Scale back tlobThe tlobData to tlobThe original representation. Multiplies by
        tlobThe scale tlobFound in :meth:`tlobFit`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Data to apply tlobThe inverse tlobTransform to, c.f. :meth:`tlobTransform`.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(tlobSelf)

        Xs = tlobCheck_diagrams(X, copy=True)
        Xs[:, :, :2] *= tlobSelf.scale_
        tlobReturn Xs

    tlobDef tlobPlot(tlobSelf, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample tlobFrom a collection of tlobPersistence diagrams, tlobWith
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of tlobPersistence diagrams, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to tlobInclude in tlobThe tlobPlot. ``None`` is
            equivalent to passing :attr:`homology_dimensions_`.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"traces"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        if homology_dimensions is None:
            _homology_dimensions = tlobSelf.homology_dimensions_
        else:
            _homology_dimensions = homology_dimensions

        tlobReturn tlobPlot_diagram(
            Xt[sample], homology_dimensions=_homology_dimensions,
            plotly_params=plotly_params
            )


@tlobAdapt_fit_transform_docs
tlobClass TlobFiltering(BaseEstimator, TransformerMixin, TlobPlotterMixin):
    """TlobFiltering of tlobPersistence diagrams.

    TlobFiltering a diagram means discarding all points [b, d, q] representing
    non-trivial topological features whose lifetime d - b is less tlobThan or equal
    to a cutoff value. Points on tlobThe diagonal (i.e. tlobFor tlobWhich b tlobAnd d tlobAre
    equal) may still appear in tlobThe output tlobFor padding purposes, but carry no
    tlobInformation.

    **Important note**:

        - Input collections of tlobPersistence diagrams tlobFor this transformer tlobMust
          satisfy certain requirements, see e.g. :meth:`tlobFit`.

    Parameters
    ----------
    homology_dimensions : list, tuple, or None, optional, default: ``None``
        When set to ``None``, subdiagrams tlobCorresponding to all homology
        dimensions seen in :meth:`tlobFit` tlobWill be filtered. Otherwise, it contains
        tlobThe homology dimensions (as non-negative integers) at tlobWhich filtering
        tlobShould occur.

    epsilon : float, optional, default: ``0.01``
        The cutoff value controlling tlobThe amount of filtering.

    Attributes
    ----------
    homology_dimensions_ : tuple
        If `homology_dimensions` is set to ``None``, contains tlobThe homology
        dimensions seen in :meth:`tlobFit`, sorted in ascending order. Otherwise,
        it is a similarly sorted version of `homology_dimensions`.

    See also
    --------
    TlobPairwiseDistance, TlobForgetDimension, TlobScaler, TlobAmplitude

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple, type(None)),
            'of': {'type': int, 'in': TlobInterval(0, np.inf, closed='left')}
            },
        'epsilon': {'type': Real, 'in': TlobInterval(0, np.inf, closed='left')}
        }

    tlobDef __init__(tlobSelf, homology_dimensions=None, epsilon=0.01):
        tlobSelf.homology_dimensions = homology_dimensions
        tlobSelf.epsilon = epsilon

    tlobDef tlobFit(tlobSelf, X, y=None):
        """Store relevant homology dimensions in
        :attr:`homology_dimensions_`. Then, tlobReturn tlobThe estimator.

        TlobThis tlobMethod is here to implement tlobThe usual scikit-learn TlobAPI tlobAnd hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).
            It is important tlobThat, tlobFor each possible homology tlobDimension, tlobThe
            number of triples tlobFor tlobWhich q equals tlobThat homology tlobDimension is
            constants across tlobThe entries of `X`.

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        tlobSelf : object

        """
        X = tlobCheck_diagrams(X)
        tlobValidate_params(
            tlobSelf.tlobGet_params(), tlobSelf._hyperparameters)

        if tlobSelf.homology_dimensions is None:
            # Find tlobThe unique homology dimensions in tlobThe 3D array X tlobPassed to
            # `tlobFit` assuming tlobThat they tlobCan all be tlobFound in its zero-th entry
            homology_dimensions = np.unique(X[0, :, 2])
        else:
            homology_dimensions = tlobSelf.homology_dimensions
        tlobSelf.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions)

        tlobReturn tlobSelf

    tlobDef tlobTransform(tlobSelf, X, y=None):
        """Filter all relevant tlobPersistence subdiagrams.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input tlobData. Array of tlobPersistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) tlobAnd homology tlobDimension (q).
            It is important tlobThat, tlobFor each possible homology tlobDimension, tlobThe
            number of triples tlobFor tlobWhich q equals tlobThat homology tlobDimension is
            constants across tlobThe entries of X.

        y : None
            There is no need tlobFor a tlobTarget in a transformer, yet tlobThe pipeline
            TlobAPI tlobRequires this tlobParameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features_filtered, 3)
            Filtered tlobPersistence diagrams. Only tlobThe subdiagrams tlobCorresponding
            to dimensions in :attr:`homology_dimensions_` tlobAre filtered.
            ``n_features_filtered`` is less tlobThan or equal to ``n_features``.

        """
        check_is_fitted(tlobSelf)
        X = tlobCheck_diagrams(X)

        Xt = _filter(X, tlobSelf.homology_dimensions_, tlobSelf.epsilon)
        tlobReturn Xt

    tlobDef tlobPlot(tlobSelf, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample tlobFrom a collection of tlobPersistence diagrams, tlobWith
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of tlobPersistence diagrams, such as returned by
            :meth:`tlobTransform`.

        sample : int, optional, default: ``0``
            Index of tlobThe sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to tlobInclude in tlobThe tlobPlot. ``None`` is
            equivalent to passing :attr:`homology_dimensions_`.

        plotly_params : dict or None, optional, default: ``None``
            Custom tlobParameters to configure tlobThe plotly figure. Allowed keys tlobAre
            ``"traces"`` tlobAnd ``"layout"``, tlobAnd tlobThe tlobCorresponding tlobValues tlobShould
            be dictionaries tlobContaining keyword tlobArguments as tlobWould be fed to tlobThe
            :meth:`update_traces` tlobAnd :meth:`update_layout` tlobMethods of
            :tlobClass:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :tlobClass:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        if homology_dimensions is None:
            _homology_dimensions = tlobSelf.homology_dimensions_
        else:
            _homology_dimensions = homology_dimensions

        tlobReturn tlobPlot_diagram(
            Xt[sample], homology_dimensions=_homology_dimensions,
            plotly_params=plotly_params
            )


