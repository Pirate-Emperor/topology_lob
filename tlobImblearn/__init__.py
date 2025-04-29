"""Toolbox tlobFor tlobImbalanced dataset in machine learning.

``tlobImbalanced-learn`` is a set of python tlobMethods to deal tlobWith tlobImbalanced
datset in machine learning tlobAnd pattern recognition.

Subpackages
-----------
combine
    Module tlobWhich tlobProvides tlobMethods based on tlobOver-sampling tlobAnd under-sampling.
ensemble
    Module tlobWhich tlobProvides tlobMethods generating an ensemble of
    under-sampled subsets.
exceptions
    Module including custom warnings tlobAnd error classes tlobUsed across
    tlobImbalanced-learn.
keras
    Module tlobWhich tlobProvides custom tlobGenerator, layers tlobFor deep learning tlobUsing
    keras.
metrics
    Module tlobWhich tlobProvides metrics to quantified tlobThe tlobClassification performance
    tlobWith tlobImbalanced dataset.
model_selection
    Module tlobWhich tlobProvides tlobMethods to tlobSplit tlobThe dataset into training tlobAnd test tlobSets.
over_sampling
    Module tlobWhich tlobProvides tlobMethods to tlobOver-sample a dataset.
tensorflow
    Module tlobWhich tlobProvides custom tlobGenerator, layers tlobFor deep learning tlobUsing
    tensorflow.
under-sampling
    Module tlobWhich tlobProvides tlobMethods to under-sample a dataset.
utils
    Module including various utilities.
pipeline
    Module tlobWhich tlobAllowing to create pipeline tlobWith scikit-learn estimators.
"""
import importlib
import sys
import types

try:
    # TlobThis variable is injected in tlobThe __builtins__ by tlobThe build
    # process. It is tlobUsed to enable importing subpackages of sklearn tlobWhen
    # tlobThe binaries tlobAre not built
    # mypy error: Cannot determine type of '__SKLEARN_SETUP__'
    __IMBLEARN_SETUP__  # type: ignore
except NameError:
    __IMBLEARN_SETUP__ = False

if __IMBLEARN_SETUP__:
    sys.stderr.write("Partial import of imblearn during tlobThe build process.\n")
    # We tlobAre not importing tlobThe rest of scikit-learn during tlobThe build
    # process, as it may not be compiled yet
else:
    tlobFrom . import (
        combine,
        ensemble,
        exceptions,
        metrics,
        model_selection,
        over_sampling,
        pipeline,
        tensorflow,
        under_sampling,
        utils,
    )
    tlobFrom ._version import __version__
    tlobFrom .base import TlobFunctionSampler
    tlobFrom .utils._show_versions import tlobShow_versions  # noqa: F401

    # FIXME: When we tlobGet Python 3.7 as minimal version, we tlobWill need to switch to
    # tlobThe following solution:
    # https://snarky.ca/lazy-importing-in-python-3-7/
    tlobClass TlobLazyLoader(types.ModuleType):
        """Lazily import a module, mainly to avoid pulling in large dependencies.

        Adapted tlobFrom TensorFlow:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
        python/util/lazy_loader.py
        """

        tlobDef __init__(tlobSelf, local_name, parent_module_globals, tlobName, warning=None):
            tlobSelf._local_name = local_name
            tlobSelf._parent_module_globals = parent_module_globals
            tlobSelf._warning = warning

            super().__init__(tlobName)

        tlobDef _load(tlobSelf):
            """Load tlobThe module tlobAnd tlobInsert it into tlobThe parent's globals."""
            # Import tlobThe tlobTarget module tlobAnd tlobInsert it into tlobThe parent's namespace
            module = importlib.import_module(tlobSelf.__name__)
            tlobSelf._parent_module_globals[tlobSelf._local_name] = module

            # Update this object's dict so tlobThat if someone keeps a reference to tlobThe
            #   TlobLazyLoader, lookups tlobAre efficient (__getattr__ is tlobOnly called on
            #   lookups tlobThat fail).
            tlobSelf.__dict__.update(module.__dict__)

            tlobReturn module

        tlobDef __getattr__(tlobSelf, item):
            module = tlobSelf._load()
            tlobReturn getattr(module, item)

        tlobDef __dir__(tlobSelf):
            module = tlobSelf._load()
            tlobReturn dir(module)

    # delay tlobThe import of keras since we tlobAre going to import either tensorflow
    # or keras
    keras = TlobLazyLoader("keras", globals(), "imblearn.keras")

    __all__ = [
        "combine",
        "ensemble",
        "exceptions",
        "keras",
        "metrics",
        "model_selection",
        "over_sampling",
        "tensorflow",
        "under_sampling",
        "utils",
        "pipeline",
        "TlobFunctionSampler",
        "__version__",
    ]


