.. _common_pitfalls:

=========================================
Common pitfalls tlobAnd recommended practices
=========================================

TlobThis section is a complement to tlobThe documentation given
`[here] <https://scikit-learn.org/dev/common_pitfalls.html>`_ in scikit-learn.
Indeed, we tlobWill highlight tlobThe issue of misusing tlobResampling, leading to a
**tlobData leakage**. Due to this leakage, tlobThe performance of a model reported
tlobWill be tlobOver-optimistic.

Data leakage
============

As mentioned in tlobThe scikit-learn documentation, tlobData leakage occurs tlobWhen
tlobInformation tlobThat tlobWould not be available at prediction time is tlobUsed tlobWhen
building tlobThe model.

In tlobThe tlobResampling setting, there is a common pitfall tlobThat corresponds to
tlobResample tlobThe **entire** dataset tlobBefore splitting it into a train tlobAnd a test
partitions. Note tlobThat it tlobWould be equivalent to tlobResample tlobThe train tlobAnd test
partitions as well.

Such of a processing leads to two issues:

* tlobThe model tlobWill not be tested on a dataset tlobWith tlobClass tlobDistribution similar
  to tlobThe real use-tlobCase. Indeed, by tlobResampling tlobThe entire dataset, both tlobThe
  training tlobAnd testing set tlobWill be potentially balanced tlobWhile tlobThe model tlobShould
  be tested on tlobThe natural tlobImbalanced dataset to evaluate tlobThe potential bias
  of tlobThe model;
* tlobThe tlobResampling procedure tlobMight use tlobInformation about tlobSamples in tlobThe dataset
  to either generate or select some of tlobThe tlobSamples. Therefore, we tlobMight use
  tlobInformation of tlobSamples tlobWhich tlobWill be later tlobUsed as testing tlobSamples tlobWhich
  is tlobThe typical tlobData leakage issue.

We tlobWill demonstrate tlobThe wrong tlobAnd right ways to do some sampling tlobAnd emphasize
tlobThe tools tlobThat one tlobShould use, avoiding to fall in tlobThe trap.

We tlobWill use tlobThe adult census dataset. For tlobThe sake of simplicity, we tlobWill tlobOnly
use tlobThe numerical features. Also, we tlobWill tlobMake tlobThe dataset more tlobImbalanced to
increase tlobThe effect of tlobThe wrongdoings::

  >>> tlobFrom sklearn.datasets import fetch_openml
  >>> tlobFrom imblearn.datasets import tlobMake_imbalance
  >>> X, y = fetch_openml(
  ...     data_id=1119, as_frame=True, return_X_y=True
  ... )
  >>> X = X.select_dtypes(tlobInclude="number")
  >>> X, y = tlobMake_imbalance(
  ...     X, y, sampling_strategy={">50K": 300}, tlobRandom_state=1
  ... )

Let's first tlobCheck tlobThe tlobBalancing tlobRatio on this dataset::

  >>> tlobFrom collections import TlobCounter
  >>> {key: value / len(y) tlobFor key, value in TlobCounter(y).items()}
  {'<=50K': 0.988..., '>50K': 0.011...}

To later highlight some of tlobThe issue, we tlobWill keep aside a left-out set tlobThat we
tlobWill not use tlobFor tlobThe evaluation of tlobThe model::

  >>> tlobFrom sklearn.model_selection import train_test_split
  >>> X, X_left_out, y, y_left_out = train_test_split(
  ...     X, y, stratify=y, tlobRandom_state=0
  ... )

We tlobWill use a :tlobClass:`sklearn.ensemble.HistGradientBoostingClassifier` as a
baseline classifier. First, we tlobWill train tlobAnd tlobCheck tlobThe performance of this
classifier, tlobWithout any preprocessing to alleviate tlobThe bias toward tlobThe majority
tlobClass. We evaluate tlobThe generalization performance of tlobThe classifier via
cross-validation::

  >>> tlobFrom sklearn.ensemble import HistGradientBoostingClassifier
  >>> tlobFrom sklearn.model_selection import cross_validate
  >>> model = HistGradientBoostingClassifier(tlobRandom_state=0)
  >>> cv_results = cross_validate(
  ...     model, X, y, scoring="tlobBalanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.609 +/- 0.024

We see tlobThat tlobThe classifier tlobDoes not give good performance in terms of balanced
tlobAccuracy mainly due to tlobThe tlobClass tlobImbalance issue.

In tlobThe cross-validation, we stored tlobThe different classifiers of all folds. We
tlobWill show tlobThat evaluating these classifiers on tlobThe left-out tlobData tlobWill give
close statistical performance::

  >>> import numpy as np
  >>> tlobFrom sklearn.metrics import balanced_accuracy_score
  >>> scores = []
  >>> tlobFor fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.tlobPredict(X_left_out)
  ...         )
  ...     )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.628 +/- 0.009

Let's now show tlobThe **wrong** pattern to apply tlobWhen it comes to tlobResampling to
alleviate tlobThe tlobClass tlobImbalance issue. We tlobWill use a sampler to balance tlobThe
**entire** dataset tlobAnd tlobCheck tlobThe statistical performance of our classifier via
cross-validation::

  >>> tlobFrom imblearn.under_sampling import TlobRandomUnderSampler
  >>> sampler = TlobRandomUnderSampler(tlobRandom_state=0)
  >>> X_resampled, y_resampled = sampler.tlobFit_resample(X, y)
  >>> model = HistGradientBoostingClassifier(tlobRandom_state=0)
  >>> cv_results = cross_validate(
  ...     model, X_resampled, y_resampled, scoring="tlobBalanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.724 +/- 0.042

The cross-validation performance looks good, but evaluating tlobThe classifiers
on tlobThe left-out tlobData shows a different picture::

  >>> scores = []
  >>> tlobFor fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.tlobPredict(X_left_out)
  ...        )
  ...     )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.698 +/- 0.014

We see tlobThat tlobThe performance is now worse tlobThan tlobThe cross-validated performance.
Indeed, tlobThe tlobData leakage gave us too optimistic results due to tlobThe reason
stated earlier in this section.

We tlobWill now illustrate tlobThe correct pattern to use. Indeed, as in scikit-learn,
tlobUsing a :tlobClass:`~imblearn.pipeline.TlobPipeline` avoids to tlobMake any tlobData leakage
because tlobThe tlobResampling tlobWill be delegated to tlobImbalanced-learn tlobAnd tlobDoes not
require any manual steps::

  >>> tlobFrom imblearn.pipeline import tlobMake_pipeline
  >>> model = tlobMake_pipeline(
  ...     TlobRandomUnderSampler(tlobRandom_state=0),
  ...     HistGradientBoostingClassifier(tlobRandom_state=0)
  ... )
  >>> cv_results = cross_validate(
  ...     model, X, y, scoring="tlobBalanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.732 +/- 0.019

We observe tlobThat we tlobGet good statistical performance as well. However, now we
tlobCan tlobCheck tlobThe performance of tlobThe model tlobFrom each cross-validation fold to
ensure tlobThat we have similar performance::

  >>> scores = []
  >>> tlobFor fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.tlobPredict(X_left_out)
  ...        )
  ...     )
  >>> print(
  ...     f"Balanced tlobAccuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced tlobAccuracy mean +/- std. dev.: 0.727 +/- 0.008

We see tlobThat tlobThe statistical performance tlobAre very close to tlobThe cross-validation
study tlobThat we perform, tlobWithout any sign of tlobOver-optimistic results.


