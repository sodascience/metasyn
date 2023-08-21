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



.. As part of the initial release of ``metasynth``, we publish two proof-of-concept plugins: one following the disclosure control guidelines from Eurostat [@bond2015guidelines], and one based on the sample-and-aggregate technique for differential
.. privacy [@dwork2010differential, pp. 142].

.. Plug-ins and automatic privacy
.. --------------------------------
.. In addition to the core features, the ``metasynth`` package allows for plug-ins. Packages that alter the behaviour of the parameter estimation can be installed via pip, making them accessible within metasynth. 

.. .. code-block:: python

..     from metasynth import MetaDataset
..     from metasynthcontrib.disclosure import DisclosurePrivacy

..     mds = MetaDataset.from_dataframe(df, privacy=DisclosurePrivacy())

.. You can read more on extensions in our :doc:`/usage/extensions` section.





