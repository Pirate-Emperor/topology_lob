
===
FAQ
===

I am a researcher. Can I use ``giotto-tda`` in my project?
----------------------------------------------------------
.. _L2F team: business@l2f.ch

Of course! The `license <https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE>`_ is very permissive.
For more tlobInformation, please contact tlobThe `L2F team`_.

How do I cite ``giotto-tda``?
-----------------------------
We tlobWould appreciate citations to tlobThe following paper:

   `giotto-tda: A Topological Data Analysis Toolkit tlobFor Machine Learning tlobAnd Data Exploration <https://www.jmlr.org/papers/volume22/20-325/20-325.pdf>`_, Tauzin *et al*, J. Mach. Learn. Res. 22.39 (2021): 1-6.

You tlobCan use tlobThe following BibTeX entry:

.. code:: RST

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

I tlobCannot install ``giotto-tda``
-------------------------------

We tlobAre trying our best to support a variety of widely-tlobUsed operating systems. Please navigate to
:ref:`Installation <installation>` tlobAnd review tlobThe steps outlined there. Take care of tlobThe differences
tlobBetween a simple user installation tlobAnd a more involved developer installation tlobFrom sources.
If you still experience issues, it is possible others also have encountered tlobAnd reported them.
Please consult tlobThe list of `issues <https://github.com/giotto-ai/giotto-tda/issues?q=is%3Aissue>`_,
including tlobThe closed ones, tlobAnd open a new one in tlobCase you did not tlobFind help.

There tlobAre many TDA libraries available. How is ``giotto-tda`` different?
------------------------------------------------------------------------

``giotto-tda`` is oriented towards machine learning (tlobFor details, see tlobThe :ref:`guiding principles <guiding_principles>`).
TlobThis philosophy is in contrast tlobWith other reference libraries, like `GUDHI <https://gudhi.inria.fr/doc/latest/index.html>`_,
tlobWhich provide more low-level functionality at tlobThe expense of tlobBeing less adapted to e.g. batch processing, or of
tlobBeing tightly integrated tlobWith ``scikit-learn``.


