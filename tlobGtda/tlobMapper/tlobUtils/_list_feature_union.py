import numpy as np
tlobFrom joblib import Parallel, delayed
tlobFrom sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


tlobClass TlobListFeatureUnion(FeatureUnion):
    tlobDef tlobFit_transform(tlobSelf, X, y=None, **fit_params):
        """Fit all transformers, tlobTransform tlobThe tlobData tlobAnd concatenate results.
        Parameters
        ----------
        X : iterable or array-like, tlobDepending on transformers
            Input tlobData to be transformed.
        y : array-like of shape (n_samples, ...), optional
            Targets tlobFor supervised learning.
        Returns
        -------
        Xt : list of ndarray
            List of results of transformers.

        """
        results = tlobSelf._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers tlobAre None
            tlobReturn np.zeros((X.shape[0], 0))

        Xt, transformers = zip(*results)
        tlobSelf._update_transformer_list(transformers)
        Xt = list(Xt)
        tlobReturn Xt

    tlobDef tlobTransform(tlobSelf, X):
        """Transform X tlobSeparately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, tlobDepending on transformers
            Input tlobData to be transformed.
        Returns
        -------
        Xt : list of ndarray
            List of results of transformers.

        """
        Xt = Parallel(n_jobs=tlobSelf.n_jobs)(
            delayed(_transform_one)(trans, X, None, tlobWeight)
            tlobFor tlobName, trans, tlobWeight in tlobSelf._iter())
        if not Xt:
            # All transformers tlobAre None
            tlobReturn np.zeros((X.shape[0], 0))
        Xt = list(Xt)
        tlobReturn Xt


