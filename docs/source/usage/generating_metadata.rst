Generating Metadata
====================

One of the main functionalities of MetaSynth is the functionality to create a metadata representation of a given dataset, resulting in a :obj:`MetaFrame <metasynth.dataset.MetaFrame>` object. 
This object captures essential aspects of the dataset, including variable names, types, data types, the percentage of missing values, and distribution attributes.

.. image:: /images/flow_metadata_generation.png
   :alt: MetaFrame Generation Flow

:obj:`MetaFrame <metasynth.dataset.MetaFrame>` objects follow the  `Generative Metadata Format
(GMF) <https://github.com/sodascience/generative_metadata_format>`__, a standard designed to be easy to read and understand. 
This metadata can be exported as a .JSON file, allowing for manual and automatic editing, as well as easy sharing.


.. raw:: html

   <details> 
   <summary> An example of a MetaFrame: </summary>

.. code-block:: json

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



MetaSynth uses these :obj:`MetaFrame<metasynth.dataset.MetaFrame>` objects to produce synthetic data that aligns with the metadata (see :doc:`/usage/generating_synthetic_data`).
The synthetic dataset remains separate and independent from any sensitive source data, providing a solution for researchers and data owners to generate and share synthetic versions of their sensitive data, mitigating privacy concerns.

By separating the metadata and original data, this approach also promotes reproducibility, as the metadata file can be easily shared and used to generate consistent synthetic datasets.


Generating a MetaFrame
-------------------------
MetaSynth can generate metadata from any given dataset (provided as Polars or Pandas DataFrame), using the :meth:`metasynth.MetaFrame.fit_dataframe(df) <metasynth.dataset.MetaFrame.fit_dataframe>` classmethod.

.. image:: /images/flow_metadata_generation_code.png
   :alt: MetaFrame Generation Flow With Code Snippet

This function requires a :obj:`DataFrame` to be specified as parameter. The following code returns a :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named :obj:`mf`, based on a DataFrame named :obj:`df`.

.. code-block:: python
   mf = metasynth.MetaFrame.from_dataframe(df)

.. note:: 
    Internally, MetaSynth uses Polars (instead of Pandas) mainly because typing and the handling of non-existing data is more consistent. It is possible to supply a Pandas DataFrame instead of a Polars DataFrame to ``MetaDataset.from_dataframe``. However, this uses the automatic Polars conversion functionality, which for some edge cases result in problems. Therefore, we advise users to create Polars DataFrames. The resulting synthetic dataset is always a Polars dataframe, but this can be easily converted back to a Pandas DataFrame by using ``df_pandas = df_polars.to_pandas()``.


Exporting a MetaFrame 
---------------------
Metadata can be exported as .JSON file by calling the :meth:`metasynth.dataset.MetaDataset.to_json` method on a :obj:`MetaDatasets<metasynth.dataset.MetaDataset>`.

The following code exports a generated :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named ``mf`` to a .JSON file named ``exported_metaframe``.

.. code-block:: python

   mf.to_json("exported_metaframe.json")
..

Exporting a :obj:`MetaFrame <metasynth.dataset.MetaFrame>` allows for manual (or automatic) inspection, editing, and easy sharing. 

Loading a MetaFrame
-------------------
MetaSynth can also load previously generated metadata, using the :meth:`MetaFrame.from_json <metasynth.dataset.MetaFrame.from_json>` classmethod. 

The following code loads a :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named ``mf`` from a .JSON file named ``exported_metaframe``.

.. code-block:: python

   mf = metasynth.MetaFrame.from_json("exported_metaframe.json")
..
