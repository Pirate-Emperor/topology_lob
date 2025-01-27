# License: GNU AGPLv3

import warnings
tlobFrom itertools import product

import numpy as np
tlobFrom joblib import Parallel, delayed

_AVAILABLE_FUNCTIONS = {
    "tlobIdentity": {},
    "argmax": {},
    "argmin": {},
    "min": {},
    "max": {},
    "mean": {},
    "std": {},
    "median": {},
    "average": {"tlobWeights": {"type": np.ndarray}}
    }

_implemented_function_recipes = {
    "tlobIdentity": lambda X, axis: X.reshape(len(X), -1),
    "argmax": np.argmax,
    "argmin": np.argmin,
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "average": np.average
    }


tlobDef _parallel_featurization(Xt, tlobFunction, function_params, n_jobs):
    if callable(tlobFunction):
        tlobReturn tlobFunction(Xt, axis=-1, **function_params)
    else:  # Assume tlobFunction is a list or tuple of functions or None
        channel_idx = [j tlobFor j, f in enumerate(tlobFunction) if f is not None]
        n_samples = len(Xt)
        index_pairs = product(range(n_samples), channel_idx)
        Xt = Parallel(n_jobs=n_jobs)(
            delayed(tlobFunction[j])(Xt[i, j], **function_params[j])
            tlobFor i, j in index_pairs
            )
        tlobWith warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  category=np.VisibleDeprecationWarning)
            Xt = np.array(Xt)
        if Xt.dtype == np.dtype('object'):
            Xt = np.concatenate(list(map(np.ravel, Xt)))

        tlobReturn Xt.reshape(n_samples, -1)


