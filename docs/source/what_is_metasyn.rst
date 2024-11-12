What is metasyn?
================

``Metasyn`` is a python package for generating synthetic data with a focus on maintaining privacy.
It is aimed at owners of sensitive datasets such as public organisations, research groups, and individual researchers who
want to improve the accessibility, reproducibility and reusability of their data. The goal of ``metasyn`` is to make it easy
for data owners to share the structure and approximation of the content of their data with others with fewer privacy concerns.

Use cases
---------

Below are a few possible use cases for metasyn:

- A researcher who cannot share their dataset because of privacy concerns, but shares a synthetic twin for reproducibility.
- A data provider wants to show a preview of the real data using synthetic twins of their datasets.
- Create synthetic data before you actually have real data so that the analysis scripts can be written and tested.

It is specifically **not** designed for creating highly accurate synthetic data, where all relationships between columns are reproduced.

Key features
------------
- **Easy**: Creating your first synthetic dataset shouldn't take more than 15 minutes with our :doc:`quickstart <quick_start>`.
- **Fast**: Creating synthetic data takes mere seconds for medium sized (~1000 rows) datasets.
- **Safe and Understandable Synthetic Data**: As little information as possible is retained from the original dataset, and you can inspect and understand exactly which information is used to create your synthetic data.
- **Flexible**: You can :doc:`adjust <improve_synth>` the synthetic data columns to your liking, create your own distributions, extension and more.
- **Data Type Support**: ``Metasyn`` supports generating synthetic data for a variety of common data types including ``categorical``, ``string``, ``integer``, ``float``, ``date``, ``time``, and ``datetime``.
- **Integration with Faker**: ``Metasyn`` integrates with the `faker <https://github.com/joke2k/faker>`__ package, a Python library for generating fake data such as names and emails. Allowing for more realistic synthetic data.
- **Structured String Detection**: ``Metasyn`` identifies structured strings within your dataset, which can include formatted text, codes, identifiers, or any string that follows a specific pattern.
- **Handling Unique Values**: ``Metasyn`` can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset.


For more detail on how metasyn works, see our `paper <https://github.com/sodascience/metasyn/blob/main/docs/paper/paper.pdf>`_.

.. _metaframes and GMF:

MetaFrames and GMF files
------------------------

.. image:: /images/pipeline_basic.png
   :alt: Metasyn Pipeline
   :align: center

One of the main distinguishing features of metasyn is the ability to serialize the model for
your dataset in a standardized, human understandable and machine readable format: the Generative
Metadata Format (GMF). The equivalent data object in Python is the :class:`metasyn.MetaFrame`.

.. admonition:: Why do I need the MetaFrame or GMF file?

  That goes back to the design philosphy of metasyn: you should audit the model before releasing
  any potentially private information. While metasyn will generally provide safe synthetic output,
  this cannot be guaranteed; whether something is considered privacy sensitive depends on the
  context, which metasyn does not have access to. We also generally recommend saving the GMF file
  for reproducibility purposes.

The GMF format is available in two file formats: .json and .toml. 

.. raw:: html

   <details> 
   <summary> An example of a .json GMF file [click to expand]: </summary>

.. code-block:: json

  {
      "n_rows": 5,
      "n_columns": 5,
      "provenance": {
          "created by": {
              "name": "metasyn",
              "version": "1.1.0"
          },
          "creation time": "2024-10-01T09:57:15.595769"
      },
      "vars": [
          {
              "name": "ID",
              "type": "discrete",
              "dtype": "Int64",
              "prop_missing": 0.0,
              "distribution": {
                  "implements": "core.unique_key",
                  "version": "1.0",
                  "provenance": "builtin",
                  "class_name": "UniqueKeyDistribution",
                  "unique": true,
                  "parameters": {
                      "lower": 1,
                      "consecutive": true
                  }
              },
              "creation_method": {
                  "created_by": "metasyn",
                  "unique": true
              }
          },
          {
              "name": "fruits",
              "type": "categorical",
              "dtype": "Categorical(ordering='physical')",
              "prop_missing": 0.0,
              "distribution": {
                  "implements": "core.multinoulli",
                  "version": "1.0",
                  "provenance": "builtin",
                  "class_name": "MultinoulliDistribution",
                  "unique": false,
                  "parameters": {
                      "labels": [
                          "apple",
                          "banana"
                      ],
                      "probs": [
                          0.4,
                          0.6
                      ]
                  }
              },
              "creation_method": {
                  "created_by": "metasyn"
              }
          },
          {
              "name": "B",
              "type": "discrete",
              "dtype": "Int64",
              "prop_missing": 0.0,
              "distribution": {
                  "implements": "core.uniform",
                  "version": "1.0",
                  "provenance": "builtin",
                  "class_name": "DiscreteUniformDistribution",
                  "unique": false,
                  "parameters": {
                      "lower": 1,
                      "upper": 6
                  }
              },
              "creation_method": {
                  "created_by": "metasyn",
                  "unique": false
              }
          }
      ]
  }

.. raw:: html

   </details> 
   <br>

.. raw:: html

   <details> 
   <summary> An example of a .toml GMF file [click to expand]: </summary>

.. code-block:: toml

  n_rows = 891 # Number of rows
  n_columns = 13 # Number of columns

  [provenance]
  "creation time" = "2024-10-28T13:58:04.604396"
  [provenance."created by"]
  name = "metasyn"
  version = "1.0.2.dev34+gd68929e"

  [[vars]] # Metadata for column with name PassengerId
  name = "PassengerId"
  type = "discrete"
  dtype = "Int64"
  prop_missing = 0.0 # Fraction of missing values, remaining: 891 values

  [vars.distribution]
  implements = "core.unique_key"
  version = "1.0"
  provenance = "metasyn-disclosure"
  class_name = "DisclosureUniqueKey"
  unique = true

  [vars.distribution.parameters]
  lower = 0
  consecutive = true

  # The above parameters for column 'PassengerId' were generated using disclosure control
  # with a maximum dominance of 0.5 and data aggregated into partitions of size 11
  # before any parameters of the distribution were estimated
  # The parameter(s) lower were estimated by the average of the 11 lowest or highest values.


  [vars.creation_method]
  created_by = "metasyn"
  unique = true

  [vars.creation_method.privacy]
  name = "disclosure"

  [vars.creation_method.privacy.parameters]
  partition_size = 11

  [[vars]] # Metadata for column with name Name
  name = "Name"
  type = "string"
  dtype = "String"
  prop_missing = 0.1 # Fraction of missing values, remaining: 802 values
  description = "Name of the unfortunate passenger of the titanic."

  [vars.distribution]
  implements = "core.faker"
  version = "1.0"
  provenance = "builtin"
  class_name = "FakerDistribution"
  unique = false

  [vars.distribution.parameters]
  faker_type = "name"
  locale = "en_US"

  # The above parameters for column 'Name' were manually set by the user, no data was (directly) used.


  [vars.creation_method]
  created_by = "metasyn"
  implements = "core.faker"

  [vars.creation_method.parameters]
  faker_type = "name"
  locale = "en_US"

  [vars.creation_method.privacy]
  name = "disclosure"

  [vars.creation_method.privacy.parameters]
  partition_size = 11

  [[vars]] # Metadata for column with name Sex
  name = "Sex"
  type = "string"
  dtype = "String"
  prop_missing = 0.0 # Fraction of missing values, remaining: 891 values

  [vars.distribution]
  implements = "core.multinoulli"
  version = "1.0"
  provenance = "metasyn-disclosure"
  class_name = "DisclosureMultinoulli"
  unique = false

  [vars.distribution.parameters]
  labels = ["female", "male"]
  probs = [0.35241301907968575, 0.6475869809203143]

  # The above parameters for column 'Sex' were generated using disclosure control
  # with a maximum dominance of 0.5 and data aggregated into partitions of size 11
  # before any parameters of the distribution were estimated.
  # Counts: [314 577]



  [vars.creation_method]
  created_by = "metasyn"

  [vars.creation_method.privacy]
  name = "disclosure"

  [vars.creation_method.privacy.parameters]
  partition_size = 11


.. raw:: html

   </details> 
   <br>

Metasyn supports both the .toml and .json variant equally. The advantage of the .json file is that it has better
compatibility with programming languages and for python you need to install the ``tomlkit`` library to write .toml files.
On the other hand, .toml files are generally easier to read and particularly edit by humans.
