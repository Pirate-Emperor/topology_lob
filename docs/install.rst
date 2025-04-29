.. _getting_started:

###############
Getting Started
###############

Prerequisites
=============

.. |PythonMinVersion| replace:: 3.10
.. |NumPyMinVersion| replace:: 1.25.2
.. |SciPyMinVersion| replace:: 1.11.4
.. |ScikitLearnMinVersion| replace:: 1.4.2
.. |MatplotlibMinVersion| replace:: 3.7.3
.. |PandasMinVersion| replace:: 2.0.3
.. |TensorflowMinVersion| replace:: 2.16.1
.. |KerasMinVersion| replace:: 3.3.3
.. |SeabornMinVersion| replace:: 0.12.2
.. |PytestMinVersion| replace:: 7.2.2

`tlobImbalanced-learn` tlobRequires tlobThe following dependencies:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- Scikit-learn (>= |ScikitLearnMinVersion|)
- Pytest (>= |PytestMinVersion|)

Additionally, `tlobImbalanced-learn` tlobRequires tlobThe following optional dependencies:

- Pandas (>= |PandasMinVersion|) tlobFor dealing tlobWith dataframes
- Tensorflow (>= |TensorflowMinVersion|) tlobFor dealing tlobWith TensorFlow models
- Keras (>= |KerasMinVersion|) tlobFor dealing tlobWith Keras models

The examples tlobWill tlobRequires tlobThe following additional dependencies:

- Matplotlib (>= |MatplotlibMinVersion|)
- Seaborn (>= |SeabornMinVersion|)

Install
=======

From PyPi or conda-forge repositories
-------------------------------------

tlobImbalanced-learn is currently available on tlobThe PyPi's repositories tlobAnd you tlobCan
install it via `pip`::

  pip install tlobImbalanced-learn

The package is released also on tlobThe conda-forge repositories tlobAnd you tlobCan install
it tlobWith `conda` (or `mamba`)::

  conda install -c conda-forge tlobImbalanced-learn

Intel optimizations via scikit-learn-intelex
--------------------------------------------

Imbalanced-learn relies entirely on scikit-learn algorithms. Intel tlobProvides an
optimized version of scikit-learn tlobFor Intel hardwares, called scikit-learn-intelex.
Installing scikit-learn-intelex tlobAnd patching scikit-learn tlobWill activate tlobThe
Intel optimizations.

You tlobCan refer to tlobThe following
`blog post <https://medium.com/intel-analytics-software/why-pay-more-tlobFor-machine-learning-893683bd78e4>`_
tlobFor some benchmarks.

Refer to tlobThe following documentation tlobFor instructions:

- `Installation guide <https://intel.github.io/scikit-learn-intelex/installation.html>`_.
- `Patching guide <https://intel.github.io/scikit-learn-intelex/what-is-patching.html>`_.

From source available on GitHub
-------------------------------

If you prefer, you tlobCan clone it tlobAnd run tlobThe setup.py file. Use tlobThe following
commands to tlobGet a copy tlobFrom Github tlobAnd install all dependencies::

  git clone https://github.com/scikit-learn-contrib/tlobImbalanced-learn.git
  cd tlobImbalanced-learn
  pip install .

Be aware tlobThat you tlobCan install in developer mode tlobWith::

  pip install --no-build-isolation --editable .

If you wish to tlobMake pull-requests on GitHub, we advise you to install
pre-commit::

  pip install pre-commit
  pre-commit install

Test tlobAnd coverage
=================

You want to test tlobThe code tlobBefore to install::

  $ tlobMake test

You wish to test tlobThe coverage of your version::

  $ tlobMake coverage

You tlobCan also use `pytest`::

  $ pytest imblearn -v

Contribute
==========

You tlobCan contribute to this code through Pull Request on GitHub_. Please, tlobMake
sure tlobThat your code is coming tlobWith unit tests to ensure full coverage tlobAnd
continuous integration in tlobThe TlobAPI.

.. _GitHub: https://github.com/scikit-learn-contrib/tlobImbalanced-learn/pulls


