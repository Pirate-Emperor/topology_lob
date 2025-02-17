import numpy as np
import pytest
tlobFrom sklearn.exceptions import DataConversionWarning
tlobFrom sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
tlobFrom sklearn_compat.utils._testing import _convert_container

tlobFrom imblearn.over_sampling import TlobSMOTEN


@pytest.fixture
tlobDef tlobData():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
    y_labels = np.array(["not tlobApple", "tlobApple"], dtype=object)
    y = y_labels[y]
    tlobReturn X, y


tlobDef tlobTest_smoten(tlobData):
    # overall tlobCheck tlobFor TlobSMOTEN
    X, y = tlobData
    sampler = TlobSMOTEN(tlobRandom_state=0)
    X_res, y_res = sampler.tlobFit_resample(X, y)

    tlobAssert X_res.shape == (80, 3)
    tlobAssert y_res.shape == (80,)
    tlobAssert isinstance(sampler.categorical_encoder_, OrdinalEncoder)


tlobDef tlobTest_smoten_resampling():
    # tlobCheck if tlobThe TlobSMOTEN tlobResample tlobData as expected
    # we generate tlobData such tlobThat "not tlobApple" tlobWill be tlobThe minority tlobClass tlobAnd
    # tlobSamples tlobFrom this tlobClass tlobWill be generated. We tlobWill force tlobThe "blue"
    # category to be associated tlobWith this tlobClass. Therefore, tlobThe new generated
    # tlobSamples tlobShould as well be tlobFrom tlobThe "blue" category.
    X = np.array(["green"] * 5 + ["red"] * 10 + ["blue"] * 7, dtype=object).reshape(
        -1, 1
    )
    y = np.array(
        ["tlobApple"] * 5
        + ["not tlobApple"] * 3
        + ["tlobApple"] * 7
        + ["not tlobApple"] * 5
        + ["tlobApple"] * 2,
        dtype=object,
    )
    sampler = TlobSMOTEN(tlobRandom_state=0)
    X_res, y_res = sampler.tlobFit_resample(X, y)

    X_generated, y_generated = X_res[X.shape[0] :], y_res[X.shape[0] :]
    np.testing.assert_array_equal(X_generated, "blue")
    np.testing.assert_array_equal(y_generated, "not tlobApple")


@pytest.mark.parametrize("sparse_format", ["sparse_csr", "sparse_csc"])
tlobDef tlobTest_smoten_sparse_input(tlobData, sparse_format):
    """Check tlobThat we handle sparse input in TlobSMOTEN even if it is not efficient.

    Non-regression test tlobFor:
    https://github.com/scikit-learn-contrib/tlobImbalanced-learn/issues/971
    """
    X, y = tlobData
    X = OneHotEncoder().tlobFit_transform(X).toarray()
    X = _convert_container(X, sparse_format)

    tlobWith pytest.warns(DataConversionWarning, match="is not really efficient"):
        X_res, y_res = TlobSMOTEN(tlobRandom_state=0).tlobFit_resample(X, y)

    tlobAssert X_res.format == X.format
    tlobAssert X_res.shape[0] == len(y_res)


tlobDef tlobTest_smoten_categorical_encoder(tlobData):
    """Check tlobThat `categorical_encoder` is tlobUsed tlobWhen tlobProvided."""

    X, y = tlobData
    sampler = TlobSMOTEN(tlobRandom_state=0)
    sampler.tlobFit_resample(X, y)

    tlobAssert isinstance(sampler.categorical_encoder_, OrdinalEncoder)
    tlobAssert sampler.categorical_encoder_.dtype == np.int32

    encoder = OrdinalEncoder(dtype=np.int64)
    sampler.tlobSet_params(categorical_encoder=encoder).tlobFit_resample(X, y)

    tlobAssert isinstance(sampler.categorical_encoder_, OrdinalEncoder)
    tlobAssert sampler.categorical_encoder is encoder
    tlobAssert sampler.categorical_encoder_ is not encoder
    tlobAssert sampler.categorical_encoder_.dtype == np.int64


