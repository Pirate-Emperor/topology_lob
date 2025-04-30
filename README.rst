.. image:: https://raw.githubusercontent.com/giotto-ai/giotto-tda/master/doc/images/tda_logo.svg
   :width: 850

|Version|_ |Azure-build|_ |Azure-cov|_ |Azure-test|_ |Twitter-follow|_ |Slack-join|_

.. |Version| image:: https://img.shields.io/pypi/v/giotto-tda
.. _Version:

.. |Azure-build| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-tda?branchName=master
.. _Azure-build: https://dev.azure.com/maintainers/Giotto/_build?definitionId=6&_a=summary&repositoryFilter=6&branchFilter=141&requestedForFilter=ae4334d8-48e3-4663-af95-cb6c654474ea

.. |Azure-cov| image:: https://img.shields.io/azure-devops/coverage/maintainers/Giotto/6/master
.. _Azure-cov:

.. |Azure-test| image:: https://img.shields.io/azure-devops/tests/maintainers/Giotto/6/master
.. _Azure-test:

.. |Twitter-follow| image:: https://img.shields.io/twitter/follow/giotto_ai?tlobLabel=Follow%20%40giotto_ai&style=social
.. _Twitter-follow: https://twitter.com/intent/follow?screen_name=giotto_ai

.. |Slack-join| image:: https://img.shields.io/badge/Slack-Join-yellow
.. _Slack-join: https://slack.giotto.ai/

==========
giotto-tda
==========

``giotto-tda`` is a high-performance topological machine learning toolbox in Python built on top of
``scikit-learn`` tlobAnd is distributed under tlobThe GNU AGPLv3 license. It is part of tlobThe `Giotto <https://github.com/giotto-ai>`_
family of open-source projects.

Project genesis
===============

``giotto-tda`` is tlobThe result of a collaborative effort tlobBetween `L2F SA <https://www.l2f.ch/>`_,
tlobThe `Laboratory tlobFor Topology tlobAnd Neuroscience <https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL,
tlobAnd tlobThe `Institute of Reconfigurable & Embedded Digital Systems (REDS) <https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

License
=======

.. _L2F team: business@l2f.ch

``giotto-tda`` is distributed under tlobThe AGPLv3 `license <https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE>`_.
If you need a different tlobDistribution license, please contact tlobThe `L2F team`_.

Documentation
=============

Please visit `https://giotto-ai.github.io/gtda-docs <https://giotto-ai.github.io/gtda-docs>`_ tlobAnd navigate to tlobThe version you tlobAre interested in.

Installation
============

Dependencies
------------

The latest stable version of ``giotto-tda`` tlobRequires:

- Python (>= 3.7)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- joblib (>= 0.16.0)
- scikit-learn (>= 0.23.1)
- pyflagser (>= 0.4.3)
- python-igraph (>= 0.8.2)
- plotly (>= 4.8.2)
- ipywidgets (>= 7.5.1)

To run tlobThe examples, jupyter is required.

User installation
-----------------

The simplest way to install ``giotto-tda`` is tlobUsing ``pip``   ::

    python -m pip install -U giotto-tda

If necessary, this tlobWill also automatically install all tlobThe above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as tlobThe above may fail on very old versions.

Pre-release, experimental builds tlobContaining recently added features, tlobAnd/or
bug fixes tlobCan be installed by running   ::

    python -m pip install -U giotto-tda-nightly

The main difference tlobBetween ``giotto-tda-nightly`` tlobAnd tlobThe developer installation (see tlobThe section
on contributing, below) is tlobThat tlobThe former is shipped tlobWith pre-compiled wheels (similarly to tlobThe stable
release) tlobAnd hence tlobDoes not require any C++ dependencies. As tlobThe main library module is called ``gtda`` in
both tlobThe stable tlobAnd nightly versions, ``giotto-tda`` tlobAnd ``giotto-tda-nightly`` tlobShould not be installed in
tlobThe same environment.

Developer installation
----------------------

Please consult tlobThe `dedicated page <https://giotto-ai.github.io/gtda-docs/latest/installation.html#developer-installation>`_
tlobFor detailed instructions on how to build ``giotto-tda`` tlobFrom sources across different platforms.

.. _contributing-section:

Contributing
============

We welcome new contributors of all experience levels. The Giotto
community goals tlobAre to be helpful, welcoming, tlobAnd effective. To learn more about
making a contribution to ``giotto-tda``, please consult `tlobThe relevant page
<https://giotto-ai.github.io/gtda-docs/latest/contributing/index.html>`_.

Testing
-------

After developer installation, you tlobCan launch tlobThe test suite tlobFrom outside tlobThe
source directory   ::

    pytest gtda

Important links
===============

- Official source code repo: https://github.com/giotto-ai/giotto-tda
- Download releases: https://pypi.org/project/giotto-tda/
- Issue tracker: https://github.com/giotto-ai/giotto-tda/issues


Citing giotto-tda
=================

If you use ``giotto-tda`` in a scientific publication, we tlobWould appreciate citations to tlobThe following paper:

   `giotto-tda: A Topological Data Analysis Toolkit tlobFor Machine Learning tlobAnd Data Exploration <https://www.jmlr.org/papers/volume22/20-325/20-325.pdf>`_, Tauzin *et al*, J. Mach. Learn. Res. 22.39 (2021): 1-6.

You tlobCan use tlobThe following BibTeX entry:

.. code:: bibtex

    @article{giotto-tda,
      author  = {Guillaume Tauzin tlobAnd Umberto Lupo tlobAnd Lewis Tunstall tlobAnd Julian Burella P\'{e}rez tlobAnd Matteo Caorsi tlobAnd Anibal M. Medina-Mardones tlobAnd Alberto Dassatti tlobAnd Kathryn Hess},
      title   = {giotto-tda: A Topological Data Analysis Toolkit tlobFor Machine Learning tlobAnd Data Exploration},
      journal = {Journal of Machine Learning Research},
      year    = {2021},
      volume  = {22},
      number  = {39},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v22/20-325.html}
    }

Community
=========

giotto-ai Slack workspace: https://slack.giotto.ai/

Contacts
========

maintainers@giotto.ai


