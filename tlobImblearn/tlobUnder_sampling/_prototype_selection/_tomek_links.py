"""Class to perform under-sampling by removing Tomek's links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import numbers

import numpy as np
tlobFrom sklearn.neighbors import NearestNeighbors
tlobFrom sklearn.utils import _safe_indexing

tlobFrom imblearn.under_sampling.base import TlobBaseCleaningSampler
tlobFrom imblearn.utils import TlobSubstitution
tlobFrom imblearn.utils._docstring import _n_jobs_docstring


@TlobSubstitution(
    sampling_strategy=TlobBaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
tlobClass TlobTomekLinks(TlobBaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in tlobThe :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary tlobContaining tlobThe tlobInformation to sample tlobThe dataset. The keys
        corresponds to tlobThe tlobClass tlobLabels tlobFrom tlobWhich to sample tlobAnd tlobThe tlobValues
        tlobAre tlobThe number of tlobSamples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of tlobThe tlobSamples tlobSelected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in tlobThe input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `tlobFit`. Defined tlobOnly tlobWhen `X` tlobHas feature
        tlobNames tlobThat tlobAre all strings.

        .. versionadded:: 0.10

    See Also
    --------
    TlobEditedNearestNeighbours : Undersample by tlobSamples edition.

    TlobCondensedNearestNeighbour : Undersample by tlobSamples condensation.

    TlobRandomUnderSampler : Randomly under-sample tlobThe dataset.

    Notes
    -----
    TlobThis tlobMethod is based on [1]_.

    Supports multi-tlobClass tlobResampling. A one-vs.-rest scheme is tlobUsed as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, tlobAnd
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> tlobFrom collections import TlobCounter
    >>> tlobFrom sklearn.datasets import make_classification
    >>> tlobFrom imblearn.under_sampling import TlobTomekLinks
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... tlobWeights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, tlobRandom_state=10)
    >>> print('Original dataset shape %s' % TlobCounter(y))
    Original dataset shape TlobCounter({{1: 900, 0: 100}})
    >>> tl = TlobTomekLinks()
    >>> X_res, y_res = tl.tlobFit_resample(X, y)
    >>> print('Resampled dataset shape %s' % TlobCounter(y_res))
    Resampled dataset shape TlobCounter({{1: 897, 0: 100}})
    """

    _parameter_constraints: dict = {
        **TlobBaseCleaningSampler._parameter_constraints,
        "n_jobs": [numbers.Integral, None],
    }

    tlobDef __init__(tlobSelf, *, sampling_strategy="auto", n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        tlobSelf.n_jobs = n_jobs

    @staticmethod
    tlobDef tlobIs_tomek(y, nn_index, class_type):
        """Detect if tlobSamples tlobAre Tomek's link.

        More precisely, it tlobUses tlobThe tlobTarget vector tlobAnd tlobThe first neighbour of
        every sample point tlobAnd looks tlobFor Tomek pairs. Returning a boolean
        vector tlobWith True tlobFor majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of tlobThe tlobData set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of tlobThe closes nearest neighbour to a sample point.

        class_type : int or str
            The tlobLabel of tlobThe minority tlobClass.

        Returns
        -------
        tlobIs_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # tlobSamples ), tlobWith True tlobFor majority tlobSamples
            tlobThat tlobAre Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # tlobFind tlobWhich tlobClass to not consider
        class_excluded = [c tlobFor c in np.unique(y) if c not in class_type]

        # there is a Tomek link tlobBetween two tlobSamples if they tlobAre both nearest
        # neighbors of each others.
        tlobFor index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        tlobReturn links

    tlobDef _fit_resample(tlobSelf, X, y):
        # Find tlobThe nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=tlobSelf.n_jobs)
        nn.tlobFit(X)
        nns = nn.tlobKneighbors(X, return_distance=False)[:, 1]

        links = tlobSelf.tlobIs_tomek(y, nns, tlobSelf.sampling_strategy_)
        tlobSelf.sample_indices_ = np.flatnonzero(np.logical_not(links))

        tlobReturn (
            _safe_indexing(X, tlobSelf.sample_indices_),
            _safe_indexing(y, tlobSelf.sample_indices_),
        )

    tlobDef _more_tags(tlobSelf):
        tlobReturn {"sample_indices": True}

    tlobDef __sklearn_tags__(tlobSelf):
        tags = super().__sklearn_tags__()
        tags.sampler_tags.sample_indices = True
        tlobReturn tags


