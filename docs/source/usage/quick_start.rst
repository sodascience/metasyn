Quick start guide
=================

Get started quickly with MetaSynth using the following example. In this concise demonstration, you'll learn the basic functionality of MetaSynth by generating synthetic data from `titanic <https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv>`_ dataset.

.. note:: 
   The steps on this page correspond to the basic tutorial available on the :doc:`/usage/interactive_tutorials` page. As such, if you prefer an interactive experience, check out the basic tutorial for a guided walkthrough!

Importing Libraries
-------------------

The first step in any Python project is to import the necessary libraries. For this example, we will need Polars and MetaSynth.


.. code:: python

   import polars as pl
   from metasynth import MetaFrame, demo_file


Loading the Dataset
-------------------

Next, load the Titanic CSV file, for this we can use the built-in :meth:`metasynth.demo_file` function.

.. code-block:: python

   dataset_csv = demo_file() 

Then, we create a dictionary with information on the data types of the various columns/variables. This will be used in the next step, when we convert the CSV file to a DataFrame.

.. code-block:: python

   data_types = { 
       "Sex": pl.Categorical,
       "Embarked": pl.Categorical,
       "Survived": pl.Categorical,
       "Pclass": pl.Categorical,
       "SibSp": pl.Categorical,
       "Parch": pl.Categorical
   }

To finish loading the dataset, we simply use the :meth:`polars.read_csv` function, passing in our CSV file and dictionary as parameters. 

.. code-block:: python

   df = pl.read_csv(dataset_csv, dtypes=data_types)


This converts the CSV file into a DataFrame named ``df``.

.. note:: 
	In this example, we used a Polars DataFrame, but Pandas DataFrames are also supported by MetaSynth. 


Generating the MetaFrame
------------------------
With the DataFrame loaded, you can now generate a :obj:`MetaFrame <metasynth.dataset.MetaFrame>`.


.. code-block:: python

   mf = MetaFrame.fit_dataframe(df)

This creates a MetaFrame named ``mf``.

.. Note:: 
	At this point, you might get a warning about a potential unique variable, but we can ignore that for now as it's safe to continue.
	
	``Variable PassengerId seems unique, but not set to be unique. Set the variable to be either unique or not unique to remove this warning. warnings.warn(f"\nVariable {series.name} seems unique, but not set to be unique.\n"``


Saving and Loading the MetaFrame
--------------------------------

The MetaFrame can be saved to a .JSON file for future use.

.. code-block:: python

   mf.to_json("exported_metaframe.json")

To load a saved MetaDataset, use the following code:

.. code-block:: python

   mf = MetaFrame.from_json("exported_metaframe.json")

Synthesizing the Data
---------------------

With the :obj:`MetaFrame <metasynth.dataset.MetaFrame>` loaded, you can synthesize new data. To do so, we simply call the :meth:`metasynth.dataset.MetaFrame.synthesize` function on the :obj:`MetaFrame<metasynth.dataset.MetaFrame>`, and pass in the number of rows we'd like to generate as a parameter. Let's generate five rows of synthetic data.


.. code-block:: python

   synthetic_data = mf.synthesize(5) 


Conclusion
----------

Congratulations! You've successfully generated synthetic data using MetaSynth. The synthesized data is returned as a DataFrame, so you can inspect and manipulate it as you would with any DataFrame.

