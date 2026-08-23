{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if methods %}
Methods
-------

{% for item in methods %}
{% if item not in inherited_members and item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}

{% endif %}
{% endfor %}
{% endif %}

{% if attributes %}
Attributes
----------

{% for item in attributes %}
{% if item not in inherited_members %}
.. autoattribute:: {{ objname }}.{{ item }}

{% endif %}
{% endfor %}
{% endif %}
