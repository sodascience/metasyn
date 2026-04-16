{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}

{% for item in attributes %}
.. autodata:: {{ item }}
   :no-index:
{% endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
   :no-index:
{% endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
{% for item in classes %}
   {{ item }}
{%- endfor %}

{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:
   :no-index:

{% endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}

{% for item in exceptions %}
.. autoexception:: {{ item }}
   :no-index:
{% endfor %}
{% endif %}
