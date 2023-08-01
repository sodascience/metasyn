.. MetaSynth documentation master file, created by
   sphinx-quickstart on Thu May 12 14:26:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |break| raw:: html

   <br />


.. image:: /images/logos/blue.svg
   :alt: MetaSynth_Logo

|break|

MetaSynth Documentation
=======================
Welcome to the `MetaSynth <https://github.com/sodascience/metasynth/>`_ documentation. 

MetaSynth is a Python package that generates synthetic tabular data. It extracts metadata like variable names, types, distributions, and the proportion of missing data from an input dataset. This metadata can be saved in an easy to read and easy to share `format <https://github.com/sodascience/generative_metadata_format>`_. MetaSynth then leverages the metadata to produce a new synthetic dataset that statistically resembles the original, without containing any original entries. The key capabilities include metadata extraction, synthetic data generation, distribution fitting, data type support, integration with Faker for realistic fake data, structured string detection, and handling of unique values. By separating metadata from actual data, MetaSynth enables sharing useful statistical patterns instead of private values. This improves reproducibility and confidentiality when working with sensitive datasets. The synthetic data can be used for code testing, experimentation, and documentation without risks of re-identification.

.. image:: https://img.shields.io/badge/GitHub-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasynth
   :alt: GitHub Repository Button
   :target: https://github.com/sodascience/metasynth

.. image:: https://img.shields.io/badge/GitHub-Issue_Tracker-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasynth%2Fissues
   :alt: GitHub Issue Tracker Button
   :target: https://github.com/sodascience/metasynth/issues

.. warning:: 
   MetaSynth, and this documentation, are under active development. As a result, this documentation is still heavily work in progress. If you see any errors, missing content or have a suggestion for new content, feel free to :doc:`let us know </about/contact>`! 

Documentation Outline
---------------------

.. toctree::
   :hidden:
   :maxdepth: 2

   about/about
   usage/usage
   developer/developer
   api/metasynth
   faq

This documentation is designed to help you easily navigate and find the information you need. It is organized into the following four sections:

:doc:`/about/about`
^^^^^^^^^^^^^^^^^^^
The :doc:`About Section </about/about>` provides a brief overview of MetaSynth's purpose and functionality, contact information, frequently asked questions (FAQs), and licensing details.

:doc:`/usage/usage`
^^^^^^^^^^^^^^^^^^^
The :doc:`/usage/usage` contains detailed, step-by-step instructions, as well as best practices for using the package. If you're new to MetaSynth, we recommend you start here!

:doc:`/api/metasynth`
^^^^^^^^^^^^^^^^^^^^^
The :doc:`/api/metasynth` is a technical reference for MetaSynth. Here, each function, class, and module is outlined in detail, giving you a comprehensive understanding of how the package works and how to use its various functionalities.  If, for example, you'd like to discover which parameters can be used for which function, you can find that here.

:doc:`/developer/developer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :doc:`/developer/developer` provides resources for those interested in contributing to MetaSynth's development. This section includes guidance on how to build upon the existing MetaSynth codebase, add new modules, functions, or even integrate other packages.






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

