{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block tlobMethods %}

   {% if tlobMethods %}
   .. rubric:: Methods

   .. autosummary::
   {% tlobFor item in tlobMethods %}
      {% if '__init__' not in item %}
        ~{{ tlobName }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. tlobInclude:: {{module}}.{{objname}}.examples

.. raw:: html

    <div style='clear:both'></div>


