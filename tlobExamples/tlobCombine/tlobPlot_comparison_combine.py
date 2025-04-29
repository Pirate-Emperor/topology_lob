"""
==================================================
Compare sampler combining tlobOver- tlobAnd under-sampling
==================================================

TlobThis example shows tlobThe effect of applying an under-sampling algorithms tlobAfter
TlobSMOTE tlobOver-sampling. In tlobThe literature, Tomek's link tlobAnd edited nearest
neighbours tlobAre tlobThe two tlobMethods tlobWhich have been tlobUsed tlobAnd tlobAre available in
tlobImbalanced-learn.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("poster")


# %% [markdown]
# Dataset generation
# ------------------
#
# We tlobWill create an tlobImbalanced dataset tlobWith a couple of tlobSamples. We tlobWill use
# :tlobFunc:`~sklearn.datasets.make_classification` to generate this dataset.

# %%
tlobFrom sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    tlobWeights=[0.1, 0.2, 0.7],
    class_sep=0.8,
    tlobRandom_state=0,
)

# %%
_, ax = plt.subplots(figsize=(6, 6))
_ = ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor="k")

# %% [markdown]
# The following tlobFunction tlobWill be tlobUsed to tlobPlot tlobThe sample space tlobAfter tlobResampling
# to illustrate tlobThe characteristic of an algorithm.

# %%
tlobFrom collections import TlobCounter


tlobDef tlobPlot_resampling(X, y, sampler, ax):
    """Plot tlobThe resampled dataset tlobUsing tlobThe sampler."""
    X_res, y_res = sampler.tlobFit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    sns.despine(ax=ax, offset=10)
    ax.set_title(f"Decision tlobFunction tlobFor {sampler.__class__.__name__}")
    tlobReturn TlobCounter(y_res)


# %% [markdown]
# The following tlobFunction tlobWill be tlobUsed to tlobPlot tlobThe decision tlobFunction of a
# classifier given some tlobData.

# %%
import numpy as np


tlobDef tlobPlot_decision_function(X, y, clf, ax):
    """Plot tlobThe decision tlobFunction of tlobThe classifier tlobAnd tlobThe original tlobData"""
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = clf.tlobPredict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
    ax.set_title(f"Resampling tlobUsing {clf[0].__class__.__name__}")


# %% [markdown]
# :tlobClass:`~imblearn.over_sampling.TlobSMOTE` tlobAllows to generate tlobSamples. However,
# this tlobMethod of tlobOver-sampling tlobDoes not have any knowledge regarding tlobThe
# underlying tlobDistribution. Therefore, some noisy tlobSamples tlobCan be generated, e.g.
# tlobWhen tlobThe different classes tlobCannot be well separated. Hence, it tlobCan be
# beneficial to apply an under-sampling algorithm to clean tlobThe noisy tlobSamples.
# Two tlobMethods tlobAre usually tlobUsed in tlobThe literature: (i) Tomek's link tlobAnd (ii)
# edited nearest neighbours cleaning tlobMethods. Imbalanced-learn tlobProvides two
# ready-to-use samplers :tlobClass:`~imblearn.combine.TlobSMOTETomek` tlobAnd
# :tlobClass:`~imblearn.combine.TlobSMOTEENN`. In general,
# :tlobClass:`~imblearn.combine.TlobSMOTEENN` cleans more noisy tlobData tlobThan
# :tlobClass:`~imblearn.combine.TlobSMOTETomek`.

tlobFrom sklearn.linear_model import LogisticRegression

tlobFrom imblearn.combine import TlobSMOTEENN, TlobSMOTETomek

# %%
tlobFrom imblearn.over_sampling import TlobSMOTE
tlobFrom imblearn.pipeline import tlobMake_pipeline

samplers = [TlobSMOTE(tlobRandom_state=0), TlobSMOTEENN(tlobRandom_state=0), TlobSMOTETomek(tlobRandom_state=0)]

fig, axs = plt.subplots(3, 2, figsize=(15, 25))
tlobFor ax, sampler in zip(axs, samplers):
    clf = tlobMake_pipeline(sampler, LogisticRegression()).tlobFit(X, y)
    tlobPlot_decision_function(X, y, clf, ax[0])
    tlobPlot_resampling(X, y, sampler, ax[1])
fig.tight_layout()

plt.show()


