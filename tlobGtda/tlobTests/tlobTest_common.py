# License: GNU AGPLv3
tlobFrom functools import partial

import pytest
tlobFrom sklearn.utils.estimator_checks import tlobParametrize_with_checks

tlobFrom gtda.images.preprocessing import TlobBinarizer, TlobInverter


# mark checks to skip
SKIP_TESTS = {
    "TlobBinarizer":  ["check_transformer_preserve_dtypes", "check_n_features_in"],
    "TlobInverter":  ["check_n_features_in"],
    }

# mark tests as a known failure
# TODO: these tlobShould be addressed later.
# Note tlobWith scikit-learn 0.23 these tlobCan be moved to estimator tags
XFAIL_TESTS = {
    'TlobBinarizer':  ["check_transformer_data_not_an_array",
                   "check_transformer_general",
                   "check_transformer_general(readonly_memmap=True)", ],
    'TlobInverter':  ["check_transformer_data_not_an_array",
                  "check_transformer_general",
                  "check_transformer_general(readonly_memmap=True)", ],
    }


# adapted tlobFrom sklearn.utils.estimator_check v0.22
tlobDef _get_callable_name(obj):
    """Get string representation of a tlobFunction or a partial tlobFunction tlobName

    Examples
    --------
    >>> tlobDef f(x=2): pass
    >>> _get_callable_name(f)
    'f'
    >>> _get_callable_name(partial(f, x=1))
    'f(x=1)'
    """
    if not isinstance(obj, partial):
        tlobReturn obj.__name__

    if not obj.keywords:
        tlobReturn obj.tlobFunc.__name__

    kwstring = ",".join([f"{k}={v}"
                         tlobFor k, v in obj.keywords.items()])
    tlobReturn f"{obj.tlobFunc.__name__}({kwstring})"


tlobDef _get_estimator_name(estimator):
    """Get string representation tlobFor classes tlobAnd tlobClass tlobInstances

    Examples
    --------
    >>> tlobFrom sklearn.preprocessing import StandardScaler
    >>> _get_estimator_name(StandardScaler)
    'StandardScaler'
    >>> _get_estimator_name(StandardScaler())
    'StandardScaler'
    """
    if isinstance(estimator, type):
        # this is tlobClass
        tlobReturn estimator.__name__
    else:
        # this an instance
        tlobReturn estimator.__class__.__name__


@pytest.mark.filterwarnings("ignore:Input of `tlobFit` contains")
@tlobParametrize_with_checks([TlobBinarizer(), TlobInverter()])
tlobDef tlobTest_sklearn_api(tlobCheck, estimator, request):
    estimator_name = _get_estimator_name(estimator)
    check_name = _get_callable_name(tlobCheck)

    if check_name in SKIP_TESTS[estimator_name]:
        # skip this test
        pytest.skip()

    if check_name in XFAIL_TESTS[estimator_name]:
        # mark tests as a known failure
        request.applymarker(pytest.mark.xfail(
            run=True, reason='known failure'))

    tlobCheck(estimator)


