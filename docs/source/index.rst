.. MetaSynth documentation master file, created by
   sphinx-quickstart on Thu May 12 14:26:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |break| raw:: html

   <br />


.. image:: /images/logos/blue.svg
   :alt: MetaSynth_Logo

|break|

.. image:: https://img.shields.io/badge/GitHub-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasynth
   :alt: GitHub Repository Button
   :target: https://github.com/sodascience/metasynth

.. image:: https://img.shields.io/badge/GitHub-Issue_Tracker-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasynth%2Fissues
   :alt: GitHub Issue Tracker Button
   :target: https://github.com/sodascience/metasynth/issues

|break|

MetaSynth Documentation
=======================
Welcome to the `MetaSynth <https://github.com/sodascience/metasynth/>`_ documentation. 

MetaSynth is a Python package for generating synthetic tabular data that statistically resembles a source dataset without exposing real values. It has two main functionalities:

1. First, MetaSynth can **extract metadata** including variable names, types, distributions, and missing data patterns from an input dataset. This metadata describes the overarching structure and characteristics of the dataset. This metadata can be saved in an easy to read and easy to share `format <https://github.com/sodascience/generative_metadata_format>`_.

2. Second, MetaSynth leverages this metadata to **produce new synthetic data** that maintains key statistical properties and structure of the original dataset. Instead of relying on the actual data, the synthetic data is generated from the metadata.

This approach ensures the synthetic data remains separate and independent from any sensitive source data. Researchers and data owners can use MetaSynth to generate and share synthetic versions of their sensitive datasets, mitigating privacy concerns. Furthermore, the separation of metadata from original data promotes reproducibility, as the shareable metadata can be used to generate consistent synthetic data.

.. note:: 
   For more information on MetaSynth and its features, check out :doc:`/about/what_is`

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
The :doc:`About Section </about/about>` provides a brief overview of MetaSynth's purpose and functionality, contact information, and licensing details.

:doc:`/usage/usage`
^^^^^^^^^^^^^^^^^^^
The :doc:`/usage/usage` contains detailed, step-by-step instructions, as well as best practices for using the package. If you're new to MetaSynth, we recommend you start here!

:doc:`/api/metasynth`
^^^^^^^^^^^^^^^^^^^^^
The :doc:`/api/metasynth` is a technical reference for MetaSynth. Here, each function, class, and module is outlined in detail, giving you a comprehensive understanding of how the package works and how to use its various functionalities.  If, for example, you'd like to discover which parameters can be used for which function, you can find that here.

:doc:`/developer/developer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :doc:`/developer/developer` provides resources for those interested in contributing to MetaSynth's development. This section includes guidance on how to build upon the existing MetaSynth codebase, add new modules, functions, or even integrate other packages.

:doc:`/faq`
^^^^^^^^^^^^^^^^^^^
The :doc:`/faq` contains commonly asked questions and answers about MetaSynth. 



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

