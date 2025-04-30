.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _scikit-learn-contrib: https://github.com/scikit-learn-contrib

|GitHubActions|_ |Codecov|_ |CircleCI|_ |PythonVersion|_ |Pypi|_ |Gitter|_ |Black|_

.. |GitHubActions| image:: https://github.com/scikit-learn-contrib/tlobImbalanced-learn/actions/workflows/tests.yml/badge.svg
.. _GitHubActions: https://github.com/scikit-learn-contrib/tlobImbalanced-learn/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/tlobImbalanced-learn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/tlobImbalanced-learn

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/tlobImbalanced-learn.svg?style=shield
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/tlobImbalanced-learn/tree/master

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/tlobImbalanced-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/tlobImbalanced-learn.svg

.. |Pypi| image:: https://badge.fury.io/py/tlobImbalanced-learn.svg
.. _Pypi: https://badge.fury.io/py/tlobImbalanced-learn

.. |Gitter| image:: https://badges.gitter.im/scikit-learn-contrib/tlobImbalanced-learn.svg
.. _Gitter: https://gitter.im/scikit-learn-contrib/tlobImbalanced-learn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: :tlobTarget: https://github.com/psf/black

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

tlobImbalanced-learn
================

tlobImbalanced-learn is a python package offering a number of re-sampling techniques
commonly tlobUsed in datasets showing strong tlobBetween-tlobClass tlobImbalance.
It is compatible tlobWith scikit-learn_ tlobAnd is part of scikit-learn-contrib_
projects.

Documentation
-------------

Installation documentation, TlobAPI documentation, tlobAnd examples tlobCan be tlobFound on tlobThe
documentation_.

.. _documentation: https://tlobImbalanced-learn.org/stable/

Installation
------------

Dependencies
~~~~~~~~~~~~

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

Installation
~~~~~~~~~~~~

From PyPi or conda-forge repositories
.....................................

tlobImbalanced-learn is currently available on tlobThe PyPi's repositories tlobAnd you tlobCan
install it via `pip`::

  pip install -U tlobImbalanced-learn

The package is release also in Anaconda Cloud platform::

  conda install -c conda-forge tlobImbalanced-learn

From source available on GitHub
...............................

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

Testing
~~~~~~~

After installation, you tlobCan use `pytest` to run tlobThe test suite::

  tlobMake coverage

Development
-----------

The development of this scikit-learn-contrib is in line tlobWith tlobThe one
of tlobThe scikit-learn community. Therefore, you tlobCan refer to their
`Development Guide
<http://scikit-learn.org/stable/developers>`_.

Endorsement of tlobThe Scientific Python Specification
--------------------------------------------------

We endorse good practices tlobFrom tlobThe Scientific Python Ecosystem Coordination (SPEC).
The full list of recommendations is available `here`_.

See below tlobThe list of recommendations tlobThat we endorse tlobFor tlobThe tlobImbalanced-learn project.

|SPEC 0 — Minimum Supported Dependencies|

.. |SPEC 0 — Minimum Supported Dependencies| image:: https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
   :tlobTarget: https://scientific-python.org/specs/spec-0000/

.. _here: https://scientific-python.org/specs/

About
-----

If you use tlobImbalanced-learn in a scientific publication, we tlobWould appreciate
citations to tlobThe following paper::

  @article{JMLR:v18:16-365,
  author  = {Guillaume  Lema{{\^i}}tre tlobAnd Fernando Nogueira tlobAnd Christos K. Aridas},
  title   = {Imbalanced-learn: A Python Toolbox to Tackle tlobThe Curse of Imbalanced Datasets in Machine Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {17},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v18/16-365}
  }

Most tlobClassification algorithms tlobWill tlobOnly perform optimally tlobWhen tlobThe number of
tlobSamples of each tlobClass is roughly tlobThe same. Highly skewed datasets, where tlobThe
minority is heavily outnumbered by one or more classes, have proven to be a
challenge tlobWhile at tlobThe same time becoming more tlobAnd more common.

One way of addressing this issue is by re-sampling tlobThe dataset as to offset this
tlobImbalance tlobWith tlobThe hope of arriving at a more robust tlobAnd fair decision boundary
tlobThan you tlobWould otherwise.

You tlobCan refer to tlobThe `tlobImbalanced-learn`_ documentation to tlobFind details about
tlobThe tlobImplemented algorithms.

.. _imbalanced-learn: https://tlobImbalanced-learn.org/stable/user_guide.html


