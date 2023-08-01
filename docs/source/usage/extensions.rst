Extensions
==========
MetaSynth is developed with a strong focus on extensibility, allowing developers to easily build upon its existing functionality. This page aims to document major extensions that are currently available.

.. note:: 
    Refer to our :doc:`/developer/developer` for more information on how to develop (additional functionality) for MetaSynth.

Disclosure control
------------------
`Disclosure Control <https://github.com/sodascience/metasynth-disclosure-control>`_ is a plugin developed in-house for MetaSynth.

While the base MetaSynth package is generally good at protecting privacy, it doesn't adhere to any standard level of privacy. For example, the uniform distributions in the base package will simply find the lowest and highest values in the dataset, and use those as the boundaries for the uniform distribution. In some cases the minimum and maximum values can be disclosive. That is why we have built this plugin that implements the disclosure control standard.


Additional distributions
------------------------
It is possible to add additional distribution types. Unfortunately, at this moment, no additional packages with distributions are publicly available. However, we encourage developers and users to explore the possibilities and contribute their custom distributions to enrich the MetaSynth ecosystem further. 