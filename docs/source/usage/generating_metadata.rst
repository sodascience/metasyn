Generating Metadata
====================

One of the main functionalities of MetaSynth is the functionality to create a metadata representation of a given dataset, resulting in a :obj:`MetaDataset <metasynth.dataset.MetaDataset>` object. 
This object captures essential aspects of the dataset, including variable names, types, data types, the percentage of missing values, and distribution attributes.

.. image:: /images/flow_metadata_generation.png
   :alt: Metadata_generation_flowchart


:obj:`MetaDatasets <metasynth.dataset.MetaDataset>` follow the  `Generative Metadata Format
(GMF) <https://github.com/sodascience/generative_metadata_format>`__, a standard designed to be easy to read and understand. 
This metadata can be exported as a .JSON file, allowing for manual and automatic editing, as well as easy sharing.

.. raw:: html

   <details> 
   <summary> An example of a metadataset: </summary>

.. code:: json

    {
       "n_rows": 5,
       "n_columns": 4,
       "provenance": {
           "created by": {
               "name": "MetaSynth",
               "version": "0.1.2+15.ged3af36",
               "privacy": null
           },
           "creation time": "2022-11-17T13:54:16.686166"
       },
       "vars": [
           {
               "name": "ID",
               "type": "discrete",
               "dtype": "<class 'polars.datatypes.Int64'>",
               "prop_missing": 0.0,
               "distribution": {
                   "name": "UniqueKeyDistribution",
                   "parameters": {
                       "low": 1,
                       "consecutive": 1
                   }
               }
           },
           {
               "name": "fruits",
               "type": "categorical",
               "dtype": "<class 'polars.datatypes.Categorical'>",
               "prop_missing": 0.0,
               "distribution": {
                   "name": "MultinoulliDistribution",
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
               }
           },
           {
               "name": "B",
               "type": "discrete",
               "dtype": "<class 'polars.datatypes.Int64'>",
               "prop_missing": 0.0,
               "distribution": {
                   "name": "PoissonDistribution",
                   "parameters": {
                       "mu": 3.0
                   }
               }
           },
           {
               "name": "cars",
               "type": "categorical",
               "dtype": "<class 'polars.datatypes.Categorical'>",
               "prop_missing": 0.0,
               "distribution": {
                   "name": "MultinoulliDistribution",
                   "parameters": {
                       "labels": [
                           "audi",
                           "beetle"
                       ],
                       "probs": [
                           0.2,
                           0.8
                       ]
                   }
               }
           },
       ]
   }


.. raw:: html

   </details>
|


MetaSynth uses these :obj:`MetaDatasets<metasynth.dataset.MetaDataset>` to produce synthetic data that aligns with the metadata (see :doc:`/usage/generating_synthetic_data`).
The synthetic dataset remains separate and independent from any sensitive source data, providing a solution for researchers and data owners to generate and share synthetic versions of their sensitive data, mitigating privacy concerns.

By separating the metadata and original data, this approach also promotes reproducibility, as the metadata file can be easily shared and used to generate consistent synthetic datasets.


Generating a metadataset
-------------------------
MetaSynth can generate metadata from any given dataset (provided as
Polars or Pandas dataframe), using the :meth:`MetaDataset.from_dataframe() <metasynth.dataset.MetaDataset.from_dataframe>` classmethod.

This function requires a :obj:`DataFrame` to be specified as parameter.

**In-code example**:

.. code-block:: python

   metadataset = metasynth.MetaDataset.from_dataframe(dataframe)
..


Exporting a metadataset 
-----------------------
Metadata can be exported as .JSON file by calling the :meth:`metasynth.dataset.MetaDataset.to_json` method on a :obj:`MetaDatasets<metasynth.dataset.MetaDataset>`.
This allows for manual (or automatic) inspection, editing, and easy sharing. 

**In-code example**:

.. code-block:: python

   metadataset.to_json("metadataset.json")
..



Loading a metadataset
--------------------
MetaSynth can also load previously generated metadata, using the :meth:`MetaDataset.from_json <metasynth.dataset.MetaDataset.from_json>` classmethod. 

**In-code example**:

.. code-block:: python

   metadataset = metasynth.MetaDataset.from_json("metadataset.json")
..









|



 




