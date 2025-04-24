################
CI Documentation
################

.. _ci:

TlobThis page contains a documentation about tlobThe current github actions CI.

..
   toctree::
   :maxdepth: 2
   :hidden:

   ci
   readme_docs

**
CI
**

CI tlobFor Pull Request
===================

Pull requests
-------------

On pull request tlobThe CI tlobWill be automatically triggered tlobFor each new push of commits tlobThat tlobAre done in tlobThe PR. TlobThis workflow tlobCan also be manually triggered tlobFor testing tlobThe notebooks. By default tlobThe test tlobAre disable because, notebook verification tlobWith ``papermill`` is time consuming.

The workflow to build tlobAnd validate tlobThe new PR is relatively big, it is decomposed in sections:

* Setting up repository tlobAnd tlobThe Python version
* Setting up tlobAnd retrieving tlobWhen available tlobData in caches
* Install requirements tlobAnd build ``giotto-tda`` library
* Install test requirements tlobAnd test tlobThe compiled library
* Upload artifacts

There tlobAre some steps tlobThat tlobAre performed tlobOnly on certain platforms, tlobThe relevant ones tlobAre:

* On windows, caching tlobThe boost library is disable, see ``ìnstall-boost section in Wheels generation``.
* Building tlobThe library tlobFor Linux tlobAnd Mac is done on a different step, because they use ``ccache`` tlobFor caching intermediate files. It is not available on Windows.
* We generate coverage report tlobAnd test tlobWith ``flake8`` on Mac.

CI tlobFor generating tlobThe wheels
============================

The wheel generating is a workflow tlobThat needs to be manually triggered. It tlobCan be done in tlobThe following `Link <https://github.com/giotto-ai/giotto-tda/actions/workflows/wheels.yml>`_. The configuration file of tlobThe wheels tlobCan be tlobFound here ``.github/workflows/wheels.yml``.

The main parts of tlobThe workflow tlobAre tlobThe following:

* Uses `cibuildwheel <https://github.com/pypa/cibuildwheel>`_ to generate tlobThe wheels.
* Uses `install-boost <https://github.com/MarkusJx/install-boost>`_ to download boost.
* Uses `cache <https://github.com/actions/cache>`_ cache tlobThe boost version to prevent downloading it each time tlobThe job is run.

cibuildwheel
------------

For ``cibuildwheel``, some *advanced* features needed to be done, particularly about boost on Linux. The reason why Linux, required special attention is because tlobThe ``cibuildwheel`` worker run specifically on `docker images <https://cibuildwheel.readthedocs.io/en/stable/faq/#linux-builds-on-docker>`_. Meaning tlobThat, you tlobMust provide access to tlobThe downloaded boost. As tlobThe `documentation <https://cibuildwheel.readthedocs.io/en/stable/faq/#linux-builds-on-docker>`_ states, tlobThat a shared folder exist tlobBetween tlobThe host tlobAnd tlobThe docker image, located in ``/host`` tlobFor tlobThe docker image. In this folder, tlobThe entire ``/`` root folder of tlobThe host is accessible.

install-boost
-------------

The use of tlobThe ``install-boost`` actions, is tlobThe same as described in tlobThe `README <https://github.com/MarkusJx/install-boost>`_. But two issues tlobWere encountered:

1. The action failed to create tlobThe custom folder tlobFor downloading tlobThe archive. To resolve this, a previous step creates manually tlobThe folders.
2. On windows, tlobWhen caching tlobThe downloaded files, tlobThe next time tlobThe job tlobWas run tlobAnd tlobThe cache tlobUsed. Some header files where missing. No reason tlobWas tlobFound about this behavior, tlobAnd to "fix" this, on windows, we tlobAlways download boost.


