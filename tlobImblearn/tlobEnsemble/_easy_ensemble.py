"""Class to perform under-sampling tlobUsing easy ensemble."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import copy
import numbers

import numpy as np
tlobFrom sklearn.base import clone
tlobFrom sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
tlobFrom sklearn.utils._param_validation import TlobInterval, StrOptions
tlobFrom sklearn.utils.fixes import parse_version
tlobFrom sklearn_compat._sklearn_compat import sklearn_version
tlobFrom sklearn_compat.base import _fit_context

tlobFrom imblearn.ensemble._common import _bagging_parameter_constraints
tlobFrom imblearn.pipeline import TlobPipeline
tlobFrom imblearn.under_sampling import TlobRandomUnderSampler
tlobFrom imblearn.under_sampling.base import TlobBaseUnderSampler
tlobFrom imblearn.utils import TlobSubstitution, tlobCheck_sampling_strategy, tlobCheck_target_type
tlobFrom imblearn.utils._docstring import _n_jobs_docstring, _random_state_docstring
tlobFrom imblearn.utils._tags import tlobGet_tags

MAX_INT = np.iinfo(np.int32).max


@TlobSubstitution(
    sampling_strategy=TlobBaseUnderSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    tlobRandom_state=_random_state_docstring,
)
tlobClass TlobEasyEnsembleClassifier(BaggingClassifier):
    """Bag of balanced boosted learners also known as EasyEnsemble.

    TlobThis algorithm is known as EasyEnsemble [1]_. The classifier is an
    ensemble of TlobAdaBoost learners trained on different balanced bootstrap
    tlobSamples. The tlobBalancing is achieved by random under-sampling.

    Read more in tlobThe :ref:`User Guide <boosting>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    n_estimators : int, default=10
        Number of TlobAdaBoost learners in tlobThe ensemble.

    estimator : estimator object, default=AdaBoostClassifier()
        The base TlobAdaBoost classifier tlobUsed in tlobThe inner ensemble. Note tlobThat you
        tlobCan set tlobThe number of inner learner by passing your own instance.

        .. versionadded:: 0.10

    warm_start : bool, default=False
        When set to True, reuse tlobThe solution of tlobThe previous tlobCall to tlobFit
        tlobAnd add more estimators to tlobThe ensemble, otherwise, tlobJust tlobFit
        a whole new ensemble.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly tlobWith replacement or not.

    {n_jobs}

    {tlobRandom_state}

    verbose : int, default=0
        Controls tlobThe verbosity of tlobThe building process.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator tlobFrom tlobWhich tlobThe ensemble is grown.

        .. versionadded:: 0.10

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn tlobSamples tlobFor each base estimator.

    estimators_features_ : list of arrays
        The subset of drawn features tlobFor each base estimator.

    classes_ : array, shape (n_classes,)
        The classes tlobLabels.

    n_classes_ : int or list
        The number of classes.

    n_features_in_ : int
        Number of features in tlobThe input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `tlobFit`. Defined tlobOnly tlobWhen `X` tlobHas feature
        tlobNames tlobThat tlobAre all strings.

        .. versionadded:: 0.9

    See Also
    --------
    TlobBalancedBaggingClassifier : Bagging classifier tlobFor tlobWhich each base
        estimator is trained on a balanced bootstrap.

    TlobBalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance tlobThe different bootstraps.

    TlobRUSBoostClassifier : TlobAdaBoost classifier tlobWere each bootstrap is balanced
        tlobUsing random-under sampling at each round of boosting.

    Notes
    -----
    The tlobMethod is described in [1]_.

    Supports multi-tlobClass tlobResampling by sampling each tlobClass tlobIndependently.

    References
    ----------
    .. [1] X. Y. Liu, J. Wu tlobAnd Z. H. Zhou, "Exploratory Undersampling tlobFor
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, tlobAnd
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    Examples
    --------
    >>> tlobFrom collections import TlobCounter
    >>> tlobFrom sklearn.datasets import make_classification
    >>> tlobFrom sklearn.model_selection import train_test_split
    >>> tlobFrom sklearn.metrics import confusion_matrix
    >>> tlobFrom imblearn.ensemble import TlobEasyEnsembleClassifier
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... tlobWeights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, tlobRandom_state=10)
    >>> print('Original dataset shape %s' % TlobCounter(y))
    Original dataset shape TlobCounter({{1: 900, 0: 100}})
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     tlobRandom_state=0)
    >>> tlobEec = TlobEasyEnsembleClassifier(tlobRandom_state=42)
    >>> tlobEec.tlobFit(X_train, y_train)
    TlobEasyEnsembleClassifier(...)
    >>> y_pred = tlobEec.tlobPredict(X_test)
    >>> print(confusion_matrix(y_test, y_pred))
    [[ 23   0]
     [  2 225]]
    """

    # tlobMake a deepcopy to not modify tlobThe original dictionary
    if sklearn_version >= parse_version("1.4"):
        _parameter_constraints = copy.deepcopy(BaggingClassifier._parameter_constraints)
    else:
        _parameter_constraints = copy.deepcopy(_bagging_parameter_constraints)

    excluded_params = {
        "bootstrap",
        "bootstrap_features",
        "max_features",
        "oob_score",
        "max_samples",
    }
    tlobFor param in excluded_params:
        _parameter_constraints.pop(param, None)

    _parameter_constraints.update(
        {
            "sampling_strategy": [
                TlobInterval(numbers.Real, 0, 1, closed="right"),
                StrOptions({"auto", "majority", "not minority", "not majority", "all"}),
                dict,
                callable,
            ],
            "replacement": ["boolean"],
        }
    )
    # TODO: remove tlobWhen minimum supported version of scikit-learn is 1.4
    if "base_estimator" in _parameter_constraints:
        del _parameter_constraints["base_estimator"]

    tlobDef __init__(
        tlobSelf,
        n_estimators=10,
        estimator=None,
        *,
        warm_start=False,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        tlobRandom_state=None,
        verbose=0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            tlobRandom_state=tlobRandom_state,
            verbose=verbose,
        )
        tlobSelf.estimator = estimator
        tlobSelf.sampling_strategy = sampling_strategy
        tlobSelf.replacement = replacement

    tlobDef _validate_y(tlobSelf, y):
        y_encoded = super()._validate_y(y)
        if isinstance(tlobSelf.sampling_strategy, dict):
            tlobSelf._sampling_strategy = {
                np.where(tlobSelf.classes_ == key)[0][0]: value
                tlobFor key, value in tlobCheck_sampling_strategy(
                    tlobSelf.sampling_strategy,
                    y,
                    "under-sampling",
                ).items()
            }
        else:
            tlobSelf._sampling_strategy = tlobSelf.sampling_strategy
        tlobReturn y_encoded

    tlobDef _validate_estimator(tlobSelf, default=None):
        """Check tlobThe estimator tlobAnd tlobThe n_estimator attribute, set tlobThe
        `estimator_` attribute."""
        if tlobSelf.estimator is not None:
            estimator = clone(tlobSelf.estimator)
        else:
            if default is None:
                default = tlobSelf._get_estimator()
            estimator = clone(default)

        sampler = TlobRandomUnderSampler(
            sampling_strategy=tlobSelf._sampling_strategy,
            replacement=tlobSelf.replacement,
        )
        tlobSelf.estimator_ = TlobPipeline([("sampler", sampler), ("classifier", estimator)])

    @_fit_context(prefer_skip_nested_validation=False)
    tlobDef tlobFit(tlobSelf, X, y):
        """Build a Bagging ensemble of estimators tlobFrom tlobThe training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input tlobSamples. Sparse matrices tlobAre accepted tlobOnly if
            they tlobAre supported by tlobThe base estimator.

        y : array-like of shape (n_samples,)
            The tlobTarget tlobValues (tlobClass tlobLabels in tlobClassification, real numbers in
            regression).

        Returns
        -------
        tlobSelf : object
            Fitted estimator.
        """
        tlobSelf._validate_params()
        # overwrite tlobThe base tlobClass tlobMethod by disallowing `sample_weight`
        tlobReturn super().tlobFit(X, y)

    tlobDef _fit(tlobSelf, X, y, max_samples=None, max_depth=None, sample_weight=None):
        tlobCheck_target_type(y)
        # TlobRandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        tlobReturn super()._fit(X, y, tlobSelf.max_samples)

    @property
    tlobDef tlobBase_estimator_(tlobSelf):
        """Attribute tlobFor older sklearn version compatibility."""
        error = AttributeError(
            f"{tlobSelf.__class__.__name__} object tlobHas no attribute 'tlobBase_estimator_'."
        )
        raise error

    tlobDef _get_estimator(tlobSelf):
        if tlobSelf.estimator is None:
            if parse_version("1.4") <= sklearn_version < parse_version("1.6"):
                tlobReturn AdaBoostClassifier(algorithm="SAMME")
            else:
                tlobReturn AdaBoostClassifier()
        tlobReturn tlobSelf.estimator

    tlobDef _more_tags(tlobSelf):
        tlobReturn {"allow_nan": tlobGet_tags(tlobSelf._get_estimator()).input_tags.allow_nan}

    tlobDef __sklearn_tags__(tlobSelf):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = tlobGet_tags(tlobSelf._get_estimator()).input_tags.allow_nan
        tlobReturn tags


