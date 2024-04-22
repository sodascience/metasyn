.. Metasyn documentation master file, created by
   sphinx-quickstart on Thu May 12 14:26:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |break| raw:: html

   <br />


.. image:: /images/logos/blue.svg
   :alt: Metasyn_Logo

|break|

.. image:: https://img.shields.io/badge/GitHub-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasyn
   :alt: GitHub Repository Button
   :target: https://github.com/sodascience/metasyn

.. image:: https://img.shields.io/badge/GitHub-Issue_Tracker-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasyn%2Fissues
   :alt: GitHub Issue Tracker Button
   :target: https://github.com/sodascience/metasyn/issues

|break|

Metasyn Documentation
=======================
Welcome to the `metasyn <https://github.com/sodascience/metasyn/>`_ documentation. 

``Metasyn`` is a Python package for generating synthetic tabular data with a focus on privacy. It is designed for owners of sensitive datasets who want to share approximations of their data so that others can perform exploratory analysis and testing without disclosing real values.

``Metasyn`` has three main functionalities:

.. image:: /images/pipeline_basic.png
   :width: 100%
   :alt: ``Metasyn`` Pipeline
   :align: center

1. **Estimation**: ``Metasyn`` can **create a MetaFrame**, from a dataset. A MetaFrame is metadata describing a table, augmented with statistical information on the columns. It captures individual distributions and features and enables the generation of synthetic data based on it.
2. **Generation**: ``Metasyn`` can **generate synthetic data** based on a MetaFrame. The synthetic data produced solely depends on the MetaFrame, thereby maintaining a critical separation between the original sensitive data and the generated synthetic data.
3. **Serialization**: ``Metasyn`` can **export a MetaFrame** into an easy-to-read :doc:`/developer/GMF` file. This allows users to audit, understand, and modify their data generation model. These GMF files can also be imported back into Metasyn to generate synthetic data.

Researchers and data owners can use ``metasyn`` to generate and share synthetic versions of their sensitive datasets, mitigating privacy concerns. Additionally, ``metasyn`` facilitates transparency and reproducibility by allowing the underlying MetaFrames to be exported and shared. Other researchers can use these to regenerate consistent synthetic datasets, validating published work without requiring sensitive data.



.. admonition:: Key Features

   -  **MetaFrame Generation**: ``Metasyn`` allows the creation of a MetaFrame from a dataset provided as a `Polars <https://pola.rs/>`_ or `Pandas <https://pandas.pydata.org/>`_ DataFrame. MetaFrames include key characteristics such as *variable names*, *data types*, *percentage of missing values*, and *distribution parameters*. 
   -  **Exporting MetaFrames**: ``Metasyn`` can export and import MetaFrames to GMF files. These are JSON files that follow the easy-to-read and understand :doc:`/developer/GMF`.
   -  **Synthetic Data Generation**: ``Metasyn`` allows for the generation of a Polars DataFrame with synthetic data that resembles the original data.
   -  **Distribution Fitting**: ``Metasyn`` allows for manual and automatic distribution fitting.
   -  **Data Type Support**: ``Metasyn`` supports generating synthetic data for a variety of common data types including ``categorical``, ``string``, ``integer``, ``float``, ``date``, ``time``, and ``datetime``.
   -  **Integration with Faker**: ``Metasyn`` integrates with the `faker <https://github.com/joke2k/faker>`__ package, a Python library for generating fake data such as names and emails. Allowing for more realistic synthetic data.
   -  **Structured String Detection**: ``Metasyn`` identifies structured strings within your dataset, which can include formatted text, codes, identifiers, or any string that follows a specific pattern.
   -  **Handling Unique Values**: ``Metasyn`` can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset.

.. admonition:: Want to know more?

   For more information on ``metasyn`` and its features, check out the :doc:`/about/about` section.


Documentation Outline
---------------------

.. toctree::
   :hidden:
   :maxdepth: 2

   metasyn_in_detail
   usage/usage
   developer/developer
   api/metasyn
   faq
   about/about


This documentation is designed to help you easily navigate and find the information you need. It is organized into the following four sections:


:doc:`metasyn_in_detail`
^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`/metasyn_in_detail` section provides an overview of metasyn's purpose and functionality.

:doc:`/usage/usage`
^^^^^^^^^^^^^^^^^^^
The :doc:`/usage/usage` contains detailed, step-by-step instructions, as well as best practices for using the package. If you're new to metasyn, we recommend you start here!

:doc:`/api/metasyn`
^^^^^^^^^^^^^^^^^^^^^
The :doc:`/api/metasyn` is a technical reference for metasyn. Here, each function, class, and module is outlined in detail, giving you a comprehensive understanding of how the package works and how to use its various functionalities.  If, for example, you'd like to discover which parameters can be used for which function, you can find that here.

:doc:`/developer/developer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :doc:`/developer/developer` provides resources for those interested in contributing to metasyn's development. This section includes guidance on how to build upon the existing ``metasyn`` codebase, add new modules, functions, or even integrate other packages.

:doc:`/faq`
^^^^^^^^^^^^^^^^^^^
The :doc:`/faq` contains commonly asked questions and answers about metasyn. 

:doc:`/about/about`
^^^^^^^^^^^^^^^^^^^
The :doc:`About Section </about/about>` provides contact information, and licensing details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
