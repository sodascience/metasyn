Quick start guide
=================

Get started quickly with ``metasyn`` using the following example. In this concise demonstration, you'll learn the basic functionality of ``metasyn`` by generating synthetic data from `titanic <https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv>`_ dataset.

.. note:: 
   A more elaborate version of this page is also available as an interactive tutorial available on the :doc:`/usage/interactive_tutorials` page.

Importing Libraries
-------------------

The first step is to import the required Python libraries. For this example, we will need Polars and metasyn.


.. code:: python

   import polars as pl
   from metasyn import MetaFrame, demo_file


Loading the Dataset
-------------------

Next, load the Titanic CSV file, for this we can use the built-in :meth:`metasyn.demo_file` function.

.. code-block:: python

   dataset_csv = demo_file() 

When loading in the DataFrame, it is important to tell Polars which columns contain categorical values, as it cannot automatically infer this. This is important, as metasyn can later read this information, and use it to generate categorical values where necessary. For this we create a dictionary with the column names that we want to specify as categorical.

.. code-block:: python

   data_types = { 
       "Sex": pl.Categorical,
       "Embarked": pl.Categorical,
   }

To finish loading the dataset, we simply use the :meth:`polars.read_csv` function, passing in our CSV file and the ``data_types`` dictionary as parameters. 

.. code-block:: python

   df = pl.read_csv(dataset_csv, schema_overrides=data_types)


This converts the CSV file into a DataFrame named ``df``.

.. note:: 
	In this example, we used a Polars DataFrame, but Pandas DataFrames are also supported by metasyn. 


The dataset should now be loaded into the DataFrame, and we can verify this by inspecting the first 5 rows of the DataFrame using the ``df.head(5)`` function (or ``print(df.head(5)`` when running from a script).  This will output the following table:

+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+
| PassengerId | Name                                                    | Sex      | Age | Parch | Ticket             | Fare    | Cabin  | Embarked | Birthday     | Board time | Married since         | all\_NA |
+=============+=========================================================+==========+=====+=======+====================+=========+========+==========+==============+============+=======================+=========+
| 1           | "Braund, Mr. Owen Harris"                               | "male"   | 22  | 0     | "A/5 21171"        | 7.25    | null   | "S"      | "1937-10-28" | "15:53:04" | "2022-08-05 04:43:34" | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+
| 2           | "Cumings, Mrs. John Bradley \(Florence Briggs Thayer\)" | "female" | 38  | 0     | "PC 17599"         | 71.2833 | "C85"  | "C"      | null         | "12:26:00" | "2022-08-07 01:56:33" | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+
| 3           | "Heikkinen, Miss. Laina"                                | "female" | 26  | 0     | "STON/O2. 3101282" | 7.925   | null   | "S"      | "1931-09-24" | "16:08:25" | "2022-08-04 20:27:37" | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+
| 4           | "Futrelle, Mrs. Jacques Heath \(Lily May Peel\)"        | "female" | 35  | 0     | "113803"           | 53.1    | "C123" | "S"      | "1936-11-30" | null       | "2022-08-07 07:05:55" | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+
| 5           | "Allen, Mr. William Henry"                              | "male"   | 35  | 0     | "373450"           | 8.05    | null   | "S"      | "1918-11-07" | "10:59:08" | "2022-08-02 15:13:34" | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+--------------+------------+-----------------------+---------+



Generating the MetaFrame
------------------------
With the DataFrame loaded, you can now generate a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>`.


.. code-block:: python

   mf = MetaFrame.fit_dataframe(df)

This creates a MetaFrame named ``mf``.

.. Note:: 
	At this point you will get a warning because metasyn detects that 'PassengerId' contains unique values, but we did not specify this column to generate only unique values. The warning is as follows:
   
	``Variable 'PassengerId' was detected to be unique, but has not explicitly been set to unique. To generate only unique values for column 'PassengerId', set unique to True. To dismiss this warning, set unique to False."``

   The page on :doc:`/usage/generating_metaframes` covers how to set unique argument in order to generate only unique values for a column, or dismiss the warning.


We can inspect the MetaFrame by simply printing it (``print(mf)``). This will produce the following output:

.. code-block:: 

   # Rows: 891
   # Columns: 13

   Column 1: "PassengerId"
   - Variable Type: discrete
   - Data Type: Int64
   - Proportion of Missing Values: 0.0000
   - Distribution:
      - Type: core.uniform
      - Provenance: builtin
      - Parameters:
         - lower: 1
         - upper: 892

   Column 2: "Name"
   # ... 



Saving and Loading the MetaFrame
--------------------------------

The MetaFrame can be saved to a JSON file for future use, to do so we simply use the :func:`~metasyn.metaframe.MetaFrame.to_json` function on the MetaFrame (which in our case is named ``mf``), and pass in the filepath as a parameter. The following code saves the MetaFrame to a JSON file named "exported_metaframe.json":

.. code-block:: python

   mf.to_json("exported_metaframe.json")

Inversely, we can load a MetaFrame from a JSON file using the :func:`~metasyn.metaframe.MetaFrame.from_json` function, passing in the filepath as a parameter. To load our previously saved MetaFrame, we use the following code:

.. code-block:: python

   mf = MetaFrame.from_json("exported_metaframe.json")

Synthesizing the Data
---------------------

With the :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` loaded, you can synthesize new data. To do so, we simply call the :meth:`~metasyn.metaframe.MetaFrame.synthesize` method on the MetaFrame, and pass in the number of rows you would like to generate as a parameter. For example, to generate five rows of synthetic data we can use: 


.. code-block:: python

   synthetic_data = mf.synthesize(5) 


We can inspect our synthesized data by printing it (``print(synthetic_data)``). This will output a table similar to the following, but with different values as it is randomly generated: 

+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+
| PassengerId | Name                               | Sex    | Age | Parch | Ticket   | Fare      | Cabin | Embarked | Birthday       | Board time   | Married since          | all\_NA |
+=============+====================================+========+=====+=======+==========+===========+=======+==========+================+==============+========================+=========+
| 19          | "Certain. Nearly."                 | "male" | 30  | 0     | "11941"  | 2.025903  | null  | "S"      | "190-1-228"    | "10:1:209"   | null                   | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+
| 795         | "Between. Nature."                 | "male" | 43  | 0     | "2067"   | 16.766045 | null  | "S"      | "19228-0-014"  | "12:507:47"  | "2022-08-01 12:558:30" | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+
| 257         | "Country. View. Evidence."         | "male" | 44  | 0     | "451553" | 3.687185  | null  | "S"      | "1937-01-110"  | "16:537:18"  | null                   | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+
| 575         | "Scene. Reason. Low. Recent."      | "male" | 34  | 1     | "8659"   | 25.834306 | null  | "S"      | "1918-004-205" | "111:137:11" | "2022-08-33 204:26:01" | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+
| 495         | "Morning. Nice. Large. Challenge." | "male" | 8   | 0     | "9582"   | 9.150979  | "G01" | "S"      | "190-017-05"   | "11:1:24"    | "2022-07-02 205:0:52"  | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+----------------+--------------+------------------------+---------+

Of course, it's easy to see some flaws with the generated dataset, such as the names, dates and times not making a lot of sense. The page on :doc:`/usage/generating_metaframes` shows how to improve the quality of the synthesized data, such as for example generating fake names using Faker, or generating proper DateTime formatted values.

Conclusion
----------

Congratulations! You've successfully generated synthetic data using metasyn. The synthesized data is returned as a DataFrame, so you can inspect and manipulate it as you would with any DataFrame.


