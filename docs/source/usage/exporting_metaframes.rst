.. |break| raw:: html

   <br />

Exporting and importing MetaFrames
===================================

MetaSynth can **export a created MetaFrame** into an easy to read JSON file, allowing users to audit, understand, modify and share their data generation model. This exported MetaFrame can be seen as metadata.

.. image:: /images/pipeline_serialization_simple.png
   :alt: MetaFrame Serialization Flow
   :align: center

Exporting a MetaFrame
----------------------
MetaFrames can be serialized and exported to a JSON file by calling the :meth:`metasynth.dataset.MetaDataset.to_json` method on a :obj:`MetaDatasets<metasynth.dataset.MetaDataset>`. 

The following code exports a generated :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named ``mf`` to a JSON file named ``exported_metaframe``.

.. code-block:: python

   mf.to_json("exported_metaframe.json")

Exported :obj:`MetaFrame <metasynth.dataset.MetaFrame>` objects follow the  `Generative Metadata Format
(GMF) <https://github.com/sodascience/generative_metadata_format>`__, a standard designed to be easy to read and understand. 

The following is an example of an exported MetaFrame:

.. raw:: html

   <details> 
   <summary> An example of an exported MetaFrame: </summary>

.. code-block:: json
    
    {
        "n_rows": 5,
        "n_columns": 5,
        "provenance": {
            "created by": {
                "name": "MetaSynth",
                "version": "0.4.0"
            },
            "creation time": "2023-08-07T12:14:06.232957"
        },
        "vars": [
            {
                "name": "ID",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.unique_key",
                    "provenance": "builtin",
                    "class_name": "UniqueKeyDistribution",
                    "parameters": {
                        "low": 1,
                        "consecutive": 1
                    }
                }
            },
            {
                "name": "fruits",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
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
                "dtype": "Int64",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.poisson",
                    "provenance": "builtin",
                    "class_name": "PoissonDistribution",
                    "parameters": {
                        "mu": 3.0
                    }
                }
            },
            {
                "name": "cars",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
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
            {
                "name": "optional",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.2,
                "distribution": {
                    "implements": "core.discrete_uniform",
                    "provenance": "builtin",
                    "class_name": "DiscreteUniformDistribution",
                    "parameters": {
                        "low": -30,
                        "high": 301
                    }
                }
            }
        ]
    }


.. raw:: html

       </details>

|break|

Loading a MetaFrame
-------------------
You can load a MetaFrame JSON file using the :meth:`MetaFrame.from_json <metasynth.dataset.MetaFrame.from_json>` classmethod. 

The following code loads a :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named ``mf`` from a JSON file named ``exported_metaframe``.

.. code-block:: python

   mf = metasynth.MetaFrame.from_json("exported_metaframe.json")


Tweaking an exported MetaFrame
-----------------------------------
Since the JSON is formatted in an easy to read way (for both humans *and* computers), it is easy to manually edit the metadata, or to automatically edit the metadata using a script. 

For example, you can:

* Change variable names
* Add or remove variables
* Change variable types
* Modify distribution parameters
* Adjust missing data rates

Let's say we use the exported MetaFrame as provided above to synthesize 5 rows of data, resulting in the following dataset:

.. list-table::
   :widths: 10 20 10 20 20
   :header-rows: 1

   * - ID (i64)
     - fruits (cat)
     - B (i64)
     - cars (cat)
     - optional (i64)
   * - 1
     - apple
     - 1
     - beetle
     - 287
   * - 2
     - banana
     - 2
     - beetle
     - 265
   * - 3
     - apple
     - 6
     - beetle
     - 152
   * - 4
     - banana
     - 0
     - beetle
     - null
   * - 5
     - banana
     - 5
     - audi
     - 87

Well, what if we wanted to change the distribution of the ``fruits`` variable to instead be 30% ``apple``, 30% ``banana``, and introduce a new fruit ``orange`` with a distribution of 40%? We can do this by editing the ``probs`` and ``labels`` attributes of the ``fruits`` variable in the exported MetaFrame. The following is the edited MetaFrame:


.. tab:: JSON file before

    .. code-block:: json

        // ...
        {
                "name": "fruits",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
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
            // ...

.. tab:: JSON file after
    
    .. code-block:: json
        :emphasize-lines: 15, 18, 19, 20

        // ...
        {
                "name": "fruits",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
                    "parameters": {
                        "labels": [
                            "apple",
                            "banana",
                            "orange"
                        ],
                        "probs": [
                            0.3,
                            0.3,
                            0.4
                        ]
                    }
                }
            },
            // ...


If we now synthesize five rows of data using the edited MetaFrame, we get the following dataset, which as you can see contains the added ``orange`` fruit, and follows the new distribution:

.. list-table::
   :widths: 10 20 10 20 20
   :header-rows: 1

   * - ID (i64)
     - fruits (cat)
     - B (i64)
     - cars (cat)
     - optional (i64)
   * - 1
     - orange
     - 4
     - beetle
     - 208
   * - 2
     - banana
     - 1
     - beetle
     - 78
   * - 3
     - orange
     - 3
     - audi
     - -30
   * - 4
     - apple
     - 2
     - beetle
     - 164
   * - 5
     - orange
     - 5
     - audi
     - 51


As you can see, it is very easy to modify the metadata to change how data is synthesized. Similarly to this example, any other aspect of the MetaFrame can be edited, including the variable names, the variable types, the data types, the percentage of missing values, and the distribution attributes. 

.. warning:: 
    Be extra careful when manually editing JSON files as errors in names, values, or formatting can cause problems. 

