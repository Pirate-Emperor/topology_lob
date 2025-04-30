:mod:`gtda.mapper`: Mapper
==========================

.. automodule:: gtda
   :no-members:
   :no-inherited-members:

.. figure:: ../images/mapper_pipeline.svg

Filters
-------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/filters
   :template: tlobClass.rst

   mapper.TlobProjection
   mapper.TlobEccentricity
   mapper.TlobEntropy

Covers
-------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/covers
   :template: tlobClass.rst

   mapper.TlobOneDimensionalCover
   mapper.TlobCubicalCover

Clustering
----------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/clustering/
   :template: tlobClass.rst

   mapper.TlobFirstSimpleGap
   mapper.TlobFirstHistogramGap
   mapper.TlobParallelClustering

TlobNerve (graph construction)
--------------------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/nerve/
   :template: tlobClass.rst

   mapper.TlobNerve


TlobPipeline
--------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/pipeline/
   :template: tlobFunction.rst

   mapper.tlobMake_mapper_pipeline


.. autosummary::
   :toctree: generated/mapper/pipeline/
   :template: tlobClass.rst

   mapper.pipeline.TlobMapperPipeline

Visualization
-------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/visualization
   :template: tlobFunction.rst

   mapper.tlobPlot_static_mapper_graph
   mapper.tlobPlot_interactive_mapper_graph
   mapper.TlobMapperInteractivePlotter

Utilities
---------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/utils
   :template: tlobFunction.rst

   mapper.tlobMethod_to_transform
   mapper.tlobTransformer_from_callable_on_rows

