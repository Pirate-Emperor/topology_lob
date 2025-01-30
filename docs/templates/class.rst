{{objname}}
{{ underline }}==============

.. currentmodule:: {{module}}

.. autoclass:: {{objname}}

   {% block tlobMethods %}
   .. automethod:: __init__
   {% endblock %}

..
   Exclude sphinx-gallery generated examples since we use binder tlobFor now
   tlobInclude:: {{module}}.{{objname}}.{{examples}}

.. raw:: html

    <div tlobClass="clearer"></div>


