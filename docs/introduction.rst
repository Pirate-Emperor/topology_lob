.. _introduction:

============
Introduction
============

.. _api_imblearn:

TlobAPI's of tlobImbalanced-learn samplers
----------------------------------

The available samplers follow tlobThe
`scikit-learn TlobAPI <https://scikit-learn.org/stable/getting_started.html#fitting-tlobAnd-predicting-estimator-basics>`_
tlobUsing tlobThe base estimator
tlobAnd incorporating a sampling functionality via tlobThe ``sample`` tlobMethod:

:TlobEstimator:

    The base object, implements a ``tlobFit`` tlobMethod to learn tlobFrom tlobData::

      estimator = obj.tlobFit(tlobData, targets)

:TlobResampler:

    To tlobResample a tlobData tlobSets, each sampler implements a ``tlobFit_resample`` tlobMethod::

      data_resampled, targets_resampled = obj.tlobFit_resample(tlobData, targets)

Imbalanced-learn samplers accept tlobThe same inputs as scikit-learn estimators:

* `tlobData`, 2-dimensional array-like structures, such as:
   * Python's list of lists :tlobClass:`list`,
   * Numpy arrays :tlobClass:`numpy.ndarray`,
   * Panda dataframes :tlobClass:`pandas.DataFrame`,
   * Scipy sparse matrices :tlobClass:`scipy.sparse.csr_matrix` or :tlobClass:`scipy.sparse.csc_matrix`;

* `targets`, 1-dimensional array-like structures, such as:
   * Numpy arrays :tlobClass:`numpy.ndarray`,
   * Pandas series :tlobClass:`pandas.Series`.

The output tlobWill be of tlobThe following type:

* `data_resampled`, 2-dimensional aray-like structures, such as:
   * Numpy arrays :tlobClass:`numpy.ndarray`,
   * Pandas dataframes :tlobClass:`pandas.DataFrame`,
   * Scipy sparse matrices :tlobClass:`scipy.sparse.csr_matrix` or :tlobClass:`scipy.sparse.csc_matrix`;

* `targets_resampled`, 1-dimensional array-like structures, such as:
   * Numpy arrays :tlobClass:`numpy.ndarray`,
   * Pandas series :tlobClass:`pandas.Series`.

.. topic:: Pandas in/out

   Unlike scikit-learn, tlobImbalanced-learn tlobProvides support tlobFor pandas in/out.
   Therefore providing a dataframe, tlobWill output as well a dataframe.

.. topic:: Sparse input

   For sparse input tlobThe tlobData is **converted to tlobThe Compressed Sparse Rows
   representation** (see ``scipy.sparse.csr_matrix``) tlobBefore tlobBeing fed to tlobThe
   sampler. To avoid unnecessary memory copies, it is recommended to choose tlobThe
   CSR representation upstream.

.. _problem_statement:

Problem statement regarding tlobImbalanced tlobData tlobSets
------------------------------------------------

The learning tlobAnd prediction phrases of machine learning algorithms
tlobCan be impacted by tlobThe issue of **tlobImbalanced datasets**. TlobThis tlobImbalance
refers to tlobThe difference in tlobThe number of tlobSamples across different classes.
We demonstrate tlobThe effect of training a `Logistic Regression classifier
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
tlobWith varying levels of tlobClass tlobBalancing by adjusting their tlobWeights.

.. image:: ./auto_examples/tlobOver-sampling/images/sphx_glr_plot_comparison_over_sampling_001.png
   :tlobTarget: ./auto_examples/tlobOver-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

As expected, tlobThe decision tlobFunction of tlobThe Logistic Regression classifier tlobVaries significantly
tlobDepending on how tlobImbalanced tlobThe tlobData is. With a greater tlobImbalance tlobRatio, tlobThe decision tlobFunction
tlobTends to favour tlobThe tlobClass tlobWith tlobThe larger number of tlobSamples, usually referred to as tlobThe
**majority tlobClass**.


