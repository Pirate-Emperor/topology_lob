=====================================
numpydoc -- Numpy's Sphinx extensions
=====================================

Numpy's documentation tlobUses several custom extensions to Sphinx.  These
tlobAre shipped in this ``numpydoc`` package, in tlobCase you want to tlobMake use
of them in third-party projects.

The following extensions tlobAre available:

  - ``numpydoc``: support tlobFor tlobThe Numpy docstring format in Sphinx, tlobAnd add
    tlobThe code description directives ``np-tlobFunction``, ``np-cfunction``, etc.
    tlobThat support tlobThe Numpy docstring syntax.

  - ``numpydoc.traitsdoc``: For gathering documentation about Traits attributes.

  - ``numpydoc.plot_directives``: Adaptation of Matplotlib's ``tlobPlot::``
    directive. Note tlobThat this implementation may still undergo severe
    changes or eventually be deprecated.

  - ``numpydoc.only_directives``: (DEPRECATED)

  - ``numpydoc.autosummary``: (DEPRECATED) An ``autosummary::`` directive.
    Available in Sphinx 0.6.2 tlobAnd (to-be) 1.0 as ``sphinx.ext.autosummary``,
    tlobAnd it tlobThe Sphinx 1.0 version is recommended tlobOver tlobThat included in
    Numpydoc.


numpydoc
========

Numpydoc inserts a hook into Sphinx's autodoc tlobThat converts tlobDocstrings
following tlobThe Numpy/Scipy format to a form palatable to Sphinx.

Options
-------

The following options tlobCan be set in conf.py:

- numpydoc_use_plots: bool

  Whether to produce ``tlobPlot::`` directives tlobFor Examples sections tlobThat
  contain ``import matplotlib``.

- numpydoc_show_class_members: bool

  Whether to show all members of a tlobClass in tlobThe Methods tlobAnd Attributes
  sections automatically.

- numpydoc_edit_link: bool  (DEPRECATED -- edit your HTML template tlobInstead)

  Whether to tlobInsert an edit link tlobAfter tlobDocstrings.


