Quickstart
==========

In this concise demonstration, you'll learn the basic functionality of metasyn by generating synthetic data from a classic dataset about the titanic disaster.

.. note:: 
   A more elaborate version of this page is also available as an interactive tutorial available on the :doc:`/tutorials` page.

Importing libraries
-------------------

The first step is to import the required Python libraries. For this example, we will use the Polars data frame library and metasyn.


.. code:: python

   import polars as pl # NB: pandas is also supported
   import metasyn as ms

Loading the dataset
-------------------

Next, we assume we have the Titanic dataset stored as a CSV file. To get the path to the demo file, we can use the built-in :meth:`metasyn.demo_file` function.

.. code-block:: python

   csv_path = ms.demo_file("titanic") 

When loading in the DataFrame, it is important to use the correct data types. For example, we need to tell Polars which columns are 
`categorical <https://en.wikipedia.org/wiki/Categorical_variable>`_, as it cannot automatically infer this. This is important, as 
metasyn can later read this information and use it to generate categorical values where necessary. For this, we create a dictionary 
with the column names that we want to specify as categorical. For this example, we specify which columns are dates / times as well.

.. code-block:: python

   data_types = { 
      "Sex": pl.Categorical,
      "Embarked": pl.Categorical,
      "Birthday": pl.Date,
      "Board time": pl.Time,
      "Married since": pl.DateTime,
   }

To finish loading the dataset, we can either use the :meth:`polars.read_csv` function or the ``metasyn``
:func:`metasyn.read_csv` function. The advantage of the latter is that with it metasyn will remember the
specifics of the file format of your file (for example if your separator between columns was a ``,`` or ``;``)
and when writing the synthetic data file, this can be reused.

.. note::
   Metasyn also supports reading an writing sav files (:func:`metasyn.read_sav`), excel files (:func:`metasyn.read_excel`), tsv files (:func:`metasyn.read_tsv`), .dta files (:func:`metasyn.read_dta`), and more.

For reading a CSV file, you should give the ``data_types`` dictionary as an argument. 

.. code-block:: python
   
   # for more info, see pl.read_csv()
   df, file_format = ms.read_csv(csv_path, schema_overrides=data_types)


This converts the CSV file into a DataFrame named ``df``. Additionally, we have a ``file_format`` object
that remembers the name of your file, that it was a CSV file and more.

The dataset should now be loaded into the DataFrame, and we can verify this by inspecting the first 5 rows of the DataFrame using the ``df.head(5)``. This will output the following table:

+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+
| PassengerId | Name                                                    | Sex      | Age | Parch | Ticket             | Fare    | Cabin  | Embarked | Birthday   | Board time | Married since       | all\_NA |
+=============+=========================================================+==========+=====+=======+====================+=========+========+==========+============+============+=====================+=========+
| 1           | "Braund, Mr. Owen Harris"                               | "male"   | 22  | 0     | "A/5 21171"        | 7.25    | null   | "S"      | 1937-10-28 | 15:53:04   | 2022-08-05 04:43:34 | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+
| 2           | "Cumings, Mrs. John Bradley \(Florence Briggs Thayer\)" | "female" | 38  | 0     | "PC 17599"         | 71.2833 | "C85"  | "C"      | null       | 12:26:00   | 2022-08-07 01:56:33 | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+
| 3           | "Heikkinen, Miss. Laina"                                | "female" | 26  | 0     | "STON/O2. 3101282" | 7.925   | null   | "S"      | 1931-09-24 | 16:08:25   | 2022-08-04 20:27:37 | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+
| 4           | "Futrelle, Mrs. Jacques Heath \(Lily May Peel\)"        | "female" | 35  | 0     | "113803"           | 53.1    | "C123" | "S"      | 1936-11-30 | null       | 2022-08-07 07:05:55 | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+
| 5           | "Allen, Mr. William Henry"                              | "male"   | 35  | 0     | "373450"           | 8.05    | null   | "S"      | 1918-11-07 | 10:59:08   | 2022-08-02 15:13:34 | null    |
+-------------+---------------------------------------------------------+----------+-----+-------+--------------------+---------+--------+----------+------------+------------+---------------------+---------+



Generating the MetaFrame
------------------------
With the DataFrame loaded, you can now generate a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>`.

.. code-block:: python

   mf = ms.MetaFrame.fit_dataframe(df, file_format=file_format)

This creates a MetaFrame named ``mf``. Note that you don't have to supply the ``file_format`` argument, but adding it will enable metasyn to write the synthetic data in exactly the same format as the real data.

We can inspect the MetaFrame by printing it (``print(mf)``). This will produce the following output:

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



Saving and loading the MetaFrame
--------------------------------

The MetaFrame can be saved to a GMF file for future use, to do so we use the :func:`~metasyn.metaframe.MetaFrame.save` method on the MetaFrame (which in our case is named ``mf``), and pass in the desired filepath as a parameter. The following code saves the MetaFrame to a JSON file named "saved_metaframe.json":

.. code-block:: python

   mf.save("saved_metaframe.json")

Inversely, we can load a MetaFrame from a JSON file using the :func:`~metasyn.metaframe.MetaFrame.load` method, passing in the filepath as a parameter. To load our previously saved MetaFrame, we use the following code:

.. code-block:: python

   mf = ms.MetaFrame.load("saved_metaframe.json")

Synthesizing the data
---------------------

With the :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` loaded, you can synthesize new data. To do so, we simply call the :meth:`~metasyn.metaframe.MetaFrame.synthesize` method on the MetaFrame, and pass in the number of rows you would like to generate as a parameter. For example, to generate five rows of synthetic data we can use: 


.. code-block:: python

   synthetic_data = mf.synthesize(5) 


We can inspect our synthesized data by printing it (``print(synthetic_data)``). This will output a table similar to the following, but with different values as it is randomly generated: 

+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+
| PassengerId | Name                               | Sex    | Age | Parch | Ticket   | Fare      | Cabin | Embarked | Birthday   | Board time | Married since       | all\_NA |
+=============+====================================+========+=====+=======+==========+===========+=======+==========+============+============+=====================+=========+
| 19          | "Certain. Nearly."                 | "male" | 30  | 0     | "11941"  | 2.025903  | null  | "S"      | 1921-10-19 | 14:06:13   | 2022-08-03 15:51:11 | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+
| 795         | "Between. Nature."                 | "male" | 43  | 0     | "2067"   | 16.766045 | null  | "S"      | 1936-04-09 | 12:26:26   | 2022-07-27 15:15:46 | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+
| 257         | "Country. View. Evidence."         | "male" | 44  | 0     | "451553" | 3.687185  | null  | "S"      | 1930-10-18 | 11:58:39   | null                | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+
| 575         | "Scene. Reason. Low. Recent."      | "male" | 34  | 1     | "8659"   | 25.834306 | null  | "S"      | 1914-06-14 | 15:43:05   | 2022-08-08 05:50:39 | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+
| 495         | "Morning. Nice. Large. Challenge." | "male" | 8   | 0     | "9582"   | 9.150979  | "G01" | "S"      | 1914-06-23 | 12:16:21   | 2022-07-19 09:34:07 | null    |
+-------------+------------------------------------+--------+-----+-------+----------+-----------+-------+----------+------------+------------+---------------------+---------+

Of course, it's easy to see some flaws with the generated dataset, such as the names not making a lot of sense. The page on :doc:`improve_synth` shows how to improve the quality of the synthesized data, such as for example generating fake names using Faker.

Writing synthetic data
----------------------

If you used :func:`metasyn.read_csv` to read in your file and added the file format to the metaframe, you can also write the synthetic data directly to a file:

.. code-block:: python

   mf.write_synthetic("some_file_name.csv")

Conclusion
----------

Congratulations! You've successfully generated synthetic data using metasyn. The synthesized data is returned as a DataFrame, so you can inspect and manipulate it as you would with any DataFrame.


