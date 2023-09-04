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

Metasynth is a Python package for generating synthetic tabular data with a focus on privacy. It is designed for owners of sensitive datasets who want to share approximations of their data for so that others can perform exploratory analysis and testing, without disclosing real values.

MetaSynth has three main functionalities:

1. **Estimation**: MetaSynth can **create a MetaFrame**, from a dataset. A MetaFrame is essentially a fitted model that characterizes the structure of the original dataset without storing actual values. It captures individual distributions and features, enabling generation of synthetic data based on these MetaFrames and can be seen as (statistical) metadata. 
2. **Serialization**: MetaSynth can **export a MetaFrame** into an easy to read JSON file, allowing users to audit, understand, and modify their data generation model. 
3. **Generation**: MetaSynth can **generate synthetic data** based on a MetaFrame. The synthetic data produced solely depends on the MetaFrame, thereby maintaining a critical separation between the original sensitive data and the synthetic data generated.

Researchers and data owners can use MetaSynth to generate and share synthetic versions of their sensitive datasets, mitigating privacy concerns. MetaSynth also promotes reproducibility by allowing MetaFrames to be exported as JSON files, as these can be shared and used to generate consistent synthetic data. 


.. image:: /images/pipeline_basic.png
   :width: 100%
   :alt: MetaSynth Pipeline
   :align: center

.. admonition:: Key Features

   -  **MetaFrame Generation**: MetaSynth allows the creation of a MetaFrame from a dataset provided as a Polars or Pandas DataFrame. MetaFrames includes key characteristics such as *variable names*, *data types*, *the percentage of missing values*, and *distribution parameters*. 
   -  **Exporting MetaFrames**: MetaSynth can export and import MetaFrames. These MetaFrames follow the according to the `Generative Metadata Format (GMF) <https://github.com/sodascience/generative_metadata_format>`__, which is a standard designed to be easy to read and understand. 
   -  **Synthetic Data Generation**: MetaSynth allows for the generation of a polars DataFrame with synthetic data that resembles the original data.
   -  **Distribution Fitting**: MetaSynth allows for manual and automatic distribution fitting.
   -  **Data Type Support**: MetaSynth supports generating synthetic data for a variety of common data types including ``categorical``, ``string``, ``integer``, ``float``, ``date``, ``time``, and ``datetime``.
   -  **Integration with Faker**: MetaSynth integrates with the `faker <https://github.com/joke2k/faker>`__ package, a Python library for generating fake data such as names and emails. Allowing for more realistic synthetic data.
   -  **Structured String Detection**: MetaSynth identifies structured strings within your dataset, which can include formatted text, codes, identifiers, or any string that follows a specific pattern.
   -  **Handling Unique Values**: MetaSynth can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset, which is crucial for generating synthetic data that maintains the characteristics of the original dataset.

.. admonition:: Want to know more?

   For more information on MetaSynth and its features, check out the :doc:`/about/about` section.


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
The :doc:`About Section </about/about>` provides an overview of MetaSynth's purpose and functionality, contact information, and licensing details.

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

