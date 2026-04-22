Multiple table generation
=========================

.. currentmodule:: metasyn.multiframe

From version 2.1 onwards, metasyn implements primary key/foreign key relations.
You might have multiple tables where one column in one table references another
column in another table. It might be important for the utility of the synthetic data that
this relationship is also present in the synthetic data.

Consider for example the very simple set of two tables with passengers and their medical data.

.. tab:: passengers.csv

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

.. tab:: medical_data.csv

    +-------------+---------------------------------------------------------+
    | PassengerId | Medical condition                                       |
    +=============+=========================================================+
    | 3           | "Healthy"                                               |
    +-------------+---------------------------------------------------------+
    | 1           | "Fever"                                                 |
    +-------------+---------------------------------------------------------+
    | 4           | "Unknown"                                               |
    +-------------+---------------------------------------------------------+

In our analysis we might want to combine the two tables; for example we might want to analyze relate the medical condition
and the age of the passenger. For this you will need to join the tables. This might pose a problem if you generate
the synthetic tables independently of each other. The synthetic ``medical_data.csv`` version might generate ``PassengerId``
values that do not occur in the ``passengers.csv`` table, while their original versions do.

Metasyn provides the :class:`metasyn.multiframe.Multiframe` class to synthesize multiple tables at once and define
relations between columns across tables.

Column relations
----------------

Metasyn implements a few different kinds of relations between columns: ``subset`` (``SUBSET OF``), ``equal`` (``EQUALS``) and
``equal_ordered`` (``EQUAL ORDERED``). There is one extra relation ``infer`` (``INFER FROM``), which signals to metasyn to attempts to
infer the relation automatically. There are two ways to define a relation between two columns: one using a string, the other
using the :class:`metasyn.multiframe.ColumnRelation` class:

.. tab:: string

    .. code-block:: python

        from metasyn.multiframe import ColumnRelation

        relation_str = "medical_data.csv[PassengerId] SUBSET OF passengers.csv[PassengerId]"
        relation = ColumnRelation.parse(relation_str)

.. tab:: direct

    .. code-block:: python

        from metasyn.multiframe import ColumnRelation, RelationType

        relation = ColumnRelation(primary_table="passengers.csv", primary_key="PassengerId",
                                  foreign_table="medical_data.csv", foreign_key="PassengerId",
                                  relation_type=RelationType.Subset)

Multiframe
----------

The :class:`metasyn.multiframe.MultiFrame` class is the equivalent of the :class:`metasyn.metaframe.MetaFrame` for
multiple tables. The class can be created directly using initialized metaframes or you can use the :meth:`metasyn.multiframe.MultiFrame.fit_dataframes` method.


.. tab:: fit dataframes

    .. code-block:: python

        from metasyn.multiframe import MultiFrame

        dfs = {"a": pl.read_csv(...), "b": pl.read_csv(...)}

        relations = ["b[passengerId] SUBSET OF a[ID]", "a[userId] <= b[userId]"]
        multi_frame = MultiFrame.fit_dataframes(dfs, relations=relations, extra_kwargs={"a": {...}, "b": {...})

.. tab:: direct initialization

    .. code-block:: python

        from metasyn.multiframe import MultiFrame

        dfs = {"a": pl.read_csv(...), "b": pl.read_csv(...)}
        mfs = {"a": MetaFrame.fit_dataframe(dfs["a"], ...), "b": MetaFrame.fit_dataframe(dfs["b"], ...)}


        relations = ["b[passengerId] SUBSET OF a[ID]", "b[userId] EQUALS a[userId]"]
        multi_frame = MultiFrame(mfs, relations=relations, dataframes=dfs)


Synthesizing multiple tables
----------------------------

Similar to the :class:`metasyn.metaframe.MetaFrame` class, the :class:`MultiFrame` class has a :meth:`MultiFrame.synthesize` method to generate
synthetic dataframes. Tables can have a different number of rows and can be set during generation of the synthetic dataset.

.. code-block:: python

    multiframe.synthesize(n={"a": 100, "b": 200})

.. note::

    In contrast to single table synthesis, you might not be able to independently set the number of rows for each
    table. For example, if one table has an "equal" relationship with another table, the two tables should have the
    same size.