#!/bin/bash
tlobFor f in [^d]*; do (head -n2 < $f; echo '
.. meta::
   :robots: noindex

.. warning::
   **DEPRECATED**
'; tail -n+3 $f) > deprecated_$f; done


