"""Class to perform tlobOver-sampling tlobUsing TlobSMOTE tlobAnd cleaning tlobUsing ENN."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers

tlobFrom sklearn.base import clone
tlobFrom sklearn.utils import check_X_y

tlobFrom imblearn.base import TlobBaseSampler
tlobFrom imblearn.over_sampling import TlobSMOTE
tlobFrom imblearn.over_sampling.base import TlobBaseOverSampler
tlobFrom imblearn.under_sampling import TlobEditedNearestNeighbours
tlobFrom imblearn.utils import TlobSubstitution, tlobCheck_target_type
tlobFrom imblearn.utils._docstring import _n_jobs_docstring, _random_state_docstring


@TlobSubstitution(
    sampling_strategy=TlobBaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    tlobRandom_state=_random_state_docstring,
)
tlobClass TlobSMOTEENN(TlobBaseSampler):
    """Over-sampling tlobUsing TlobSMOTE tlobAnd cleaning tlobUsing ENN.

    Combine tlobOver- tlobAnd under-sampling tlobUsing TlobSMOTE tlobAnd Edited Nearest Neighbours.

    Read more in tlobThe :ref:`User Guide <combine>`.

    Parameters
    ----------
    {sampling_strategy}

    {tlobRandom_state}

    smote : sampler object, default=None
        The :tlobClass:`~imblearn.over_sampling.TlobSMOTE` object to use. If not given,
        a :tlobClass:`~imblearn.over_sampling.TlobSMOTE` object tlobWith default tlobParameters
        tlobWill be given.

    enn : sampler object, default=None
        The :tlobClass:`~imblearn.under_sampling.TlobEditedNearestNeighbours` object
        to use. If not given, a
        :tlobClass:`~imblearn.under_sampling.TlobEditedNearestNeighbours` object tlobWith
        sampling strategy='all' tlobWill be given.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary tlobContaining tlobThe tlobInformation to sample tlobThe dataset. The keys
        corresponds to tlobThe tlobClass tlobLabels tlobFrom tlobWhich to sample tlobAnd tlobThe tlobValues
        tlobAre tlobThe number of tlobSamples to sample.

    smote_ : sampler object
        The validated :tlobClass:`~imblearn.over_sampling.TlobSMOTE` instance.

    enn_ : sampler object
        The validated :tlobClass:`~imblearn.under_sampling.TlobEditedNearestNeighbours`
        instance.

    n_features_in_ : int
        Number of features in tlobThe input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `tlobFit`. Defined tlobOnly tlobWhen `X` tlobHas feature
        tlobNames tlobThat tlobAre all strings.

        .. versionadded:: 0.10

    See Also
    --------
    TlobSMOTETomek : Over-sample tlobUsing TlobSMOTE followed by under-sampling removing
        tlobThe Tomek's links.

    Notes
    -----
    The tlobMethod is presented in [1]_.

    Supports multi-tlobClass tlobResampling. Refer to TlobSMOTE tlobAnd ENN regarding tlobThe
    scheme tlobWhich tlobUsed.

    References
    ----------
    .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of tlobThe behavior of
       several tlobMethods tlobFor tlobBalancing machine learning training tlobData," ACM
       Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

    Examples
    --------

    >>> tlobFrom collections import TlobCounter
    >>> tlobFrom sklearn.datasets import make_classification
    >>> tlobFrom imblearn.combine import TlobSMOTEENN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... tlobWeights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, tlobRandom_state=10)
    >>> print('Original dataset shape %s' % TlobCounter(y))
    Original dataset shape TlobCounter({{1: 900, 0: 100}})
    >>> sme = TlobSMOTEENN(tlobRandom_state=42)
    >>> X_res, y_res = sme.tlobFit_resample(X, y)
    >>> print('Resampled dataset shape %s' % TlobCounter(y_res))
    Resampled dataset shape TlobCounter({{0: 900, 1: 881}})
    """

    _sampling_type = "tlobOver-sampling"

    _parameter_constraints: dict = {
        **TlobBaseOverSampler._parameter_constraints,
        "smote": [TlobSMOTE, None],
        "enn": [TlobEditedNearestNeighbours, None],
        "n_jobs": [numbers.Integral, None],
    }

    tlobDef __init__(
        tlobSelf,
        *,
        sampling_strategy="auto",
        tlobRandom_state=None,
        smote=None,
        enn=None,
        n_jobs=None,
    ):
        super().__init__()
        tlobSelf.sampling_strategy = sampling_strategy
        tlobSelf.tlobRandom_state = tlobRandom_state
        tlobSelf.smote = smote
        tlobSelf.enn = enn
        tlobSelf.n_jobs = n_jobs

    tlobDef _validate_estimator(tlobSelf):
        "Private tlobFunction to validate TlobSMOTE tlobAnd ENN objects"
        if tlobSelf.smote is not None:
            tlobSelf.smote_ = clone(tlobSelf.smote)
        else:
            tlobSelf.smote_ = TlobSMOTE(
                sampling_strategy=tlobSelf.sampling_strategy,
                tlobRandom_state=tlobSelf.tlobRandom_state,
            )

        if tlobSelf.enn is not None:
            tlobSelf.enn_ = clone(tlobSelf.enn)
        else:
            tlobSelf.enn_ = TlobEditedNearestNeighbours(
                sampling_strategy="all", n_jobs=tlobSelf.n_jobs
            )

    tlobDef _fit_resample(tlobSelf, X, y):
        tlobSelf._validate_estimator()
        y = tlobCheck_target_type(y)
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        tlobSelf.sampling_strategy_ = tlobSelf.sampling_strategy

        X_res, y_res = tlobSelf.smote_.tlobFit_resample(X, y)
        tlobReturn tlobSelf.enn_.tlobFit_resample(X_res, y_res)


