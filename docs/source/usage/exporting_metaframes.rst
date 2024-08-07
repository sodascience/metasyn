.. |break| raw:: html

   <br />

Exporting and importing MetaFrames
===================================

Metasyn can serialize and **export a MetaFrame** into a GMF file. GMF files are JSON files that follow the :doc:`/developer/GMF` and have been designed to be easy to read and understand. This allows users to audit, understand, modify and share their data generation model with ease. 

.. image:: /images/pipeline_serialization_simple.png
   :alt: MetaFrame Serialization Flow
   :align: center

Exporting a MetaFrame
----------------------
MetaFrames can be serialized and exported to a GMF file by calling the :meth:`metasyn.metaframe.MetaFrame.to_json` method on a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>`. 

The following code exports a generated :obj:`MetaFrame<metasyn.metaframe.MetaFrame>` object named ``mf`` to a GMF file named ``exported_metaframe``.

.. code-block:: python

   mf.to_json("exported_metaframe.json")

.. raw:: html

   <details> 
   <summary> <em><b>An example of a MetaFrame that has been exported to a GMF file: </em></b></summary>

.. code-block:: json
    
    {
        "n_rows": 5,
        "n_columns": 5,
        "provenance": {
            "created by": {
                "name": "metasyn",
                "version": "0.8.1"
            },
            "creation time": "2024-04-29T12:07:36.646884"
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
            },
            {
                "name": "cars",
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
                            "audi",
                            "beetle"
                        ],
                        "probs": [
                            0.2,
                            0.8
                        ]
                    }
                },
                "creation_method": {
                    "created_by": "metasyn"
                }
            },
            {
                "name": "optional",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.2,
                "distribution": {
                    "implements": "core.uniform",
                    "version": "1.0",
                    "provenance": "builtin",
                    "class_name": "DiscreteUniformDistribution",
                    "unique": false,
                    "parameters": {
                        "lower": -30,
                        "upper": 301
                    }
                },
                "creation_method": {
                    "created_by": "metasyn"
                }
            }
        ]
    }


.. raw:: html

       </details>

|break|

    
It is possible to preview the GMF file, without having to export it. This can be done by calling the Python built-in :func:`repr <python:repr>` function on a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>` object, and printing its output.

.. code-block:: python

    gmf_preview = repr(mf)
    print(gmf_preview)

Loading a MetaFrame
-------------------
You can load a MetaFrame from a GMF file using the :meth:`MetaFrame.from_json <metasyn.metaframe.MetaFrame.from_json>` classmethod. 

The following code loads a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>` object named ``mf`` from a GMF file named ``exported_metaframe``.

.. code-block:: python

   mf = metasyn.MetaFrame.from_json("exported_metaframe.json")


Tweaking an exported MetaFrame
-----------------------------------
Since the JSON is formatted in an easy to read way (for both humans *and* computers), it is easy to manually edit the metadata, or to automatically edit the metadata using a script. 

For example, you can:

* Change variable names
* Add or remove variables
* Change variable types
* Modify distribution parameters
* Adjust missing data rates

Let's say we import a MetaFrame from the GMF (from earlier on this page) and use it to synthesize 5 rows of data. This results in the following dataset (note that the resulting dataset will be different every time you run this code, since the data is randomly generated):

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


.. tab:: GMF file before

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

.. tab:: GMF file after
    
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


If we now synthesize five rows of data based on a MetaFrame loaded from the edited GMF file, we get the following dataset, which as you can see contains the added ``orange`` fruit, and follows the new distribution:

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


As you can see, you can modify the metadata to change how data is synthesized. Similarly to this example, any other aspect of the MetaFrame can be edited, including the variable names, the variable types, the data types, the percentage of missing values, and the distribution attributes. 

.. warning:: 
    Be extra careful when manually editing GMF files as errors in names, values, or formatting can cause problems. In this case, metasyn will most likely produce JSON validation errors.

