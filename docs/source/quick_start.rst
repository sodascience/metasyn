Quick start guide
=================

The MetaSynth package contains two discrete functionalities: creating a statistical metadata file (smf) from tabular data,
and a synthetic data generator. On this page you should be able to:

- Install the package
- Prepare your data
- Create an smf file
- Read the smf file and create a synthetic dataset.

Installing MetaSynth
--------------------

First you need to create a working Python installation, with a version above or equal to Python 3.7.
If you're not familiar with Python and have never installed it you can use the following
`guide <https://docs.python-guide.org/starting/installation/>`_.

For the first official release of MetaSynth, it will be installable directly from `PyPi <https://pypi.org/>`_.
For now, the MetaSynth package can be installed using the following command in the console:


.. code-block:: console

	pip install git+https://github.com/sodascience/meta-synth.git

To test the successful installation of MetaSynth, type the following in a python console and it should not give
any errors:

.. code-block:: python

	import metasynth


Preparing the data
------------------

To prepare the data for usage with MetaSynth, it is useful to have some basic knowledge about 
`polars <https://pola-rs.github.io/polars-book/user-guide/>`_. We use the
`titanic <https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv>`_ dataset
to show the basic concepts here. To follow along, download the dataset to a folder and start Python
in the same folder.

First read the csv file with polars:

.. code-block:: python

	import polars as pl
	
	dtypes = {
	    "Sex": pl.Categorical,
	    "Embarked": pl.Categorical,
	    "Survived": pl.Categorical,
	    "Pclass": pl.Categorical,
	    "SibSp": pl.Categorical,
	    "Parch": pl.Categorical
	}
	df = pl.read_csv("titanic.csv", dtype=dtypes)

The dtypes argument is used to set the columns to their correct type. Only categorical variables
need to be manually set. Dates, times and datetimes can be parsed by polars with the `parse_dates=True` keyword argument.
There are seven differnt data types supported by MetaSynth: string, category, integer, float, date, time and datetime.


Create a statistical metadata file
----------------------------------

Currently, only the JSON file format is supported. First we need to create a MetaSynth dataset, which infers the
distributions of the polars DataFrame:

.. code-block:: python

	from metasynth import MetaDataset

	dataset = MetaDataset.from_dataframe(df)


Internally, the dataset is simply a list of all the column variables, with their statistical properties. Next, write it
to a JSON file:

.. code-block:: python

	dataset.to_json("titanic.json")

The file titanic.json should have a statistical metadata summary that can be used to generate synthetic data.


Create a synthetic dataset
--------------------------

To create the synthetic dataset we will first read the file (this is technically not necessary in this case):

.. code-block:: python

	dataset = MetaDataset.from_json("titanic.json")


From the dataset it is easy to create a synthetic dataset with e.g. 100 rows:

.. code-block:: python

	synthetic_df = dataset.synthesize(100)


More advanced uses of MetaSynth
-------------------------------

A more advanced tutorial is available on our 
`GitHub <https://github.com/sodascience/meta-synth/blob/main/examples/advanced tutorial.ipynb`>_
page.
