{% extends "!autosummary/module.rst" %}

{# This file is almost the same as the default, but adds :toctree: and :nosignatures: to
   the autosummary directives. The original can be found at
   ``sphinx/ext/autosummary/templates/autosummary/module.rst``. #}

{% block attributes %}
{% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}

{% set types = [] %}
{% for item in members %}
   {% if not item.startswith('_') and not (item in functions or item in attributes or item in exceptions) %}
      {% set _ = types.append(item) %}
   {% endif %}
{%- endfor %}

{% if types %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in types %}
      {{ item }}
   {%- endfor %}

{% endif %}
{% endblock %}
