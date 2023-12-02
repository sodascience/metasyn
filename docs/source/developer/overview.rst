Overview
========

This page aims to introduce the main classes and modules of the ``metasyn`` package. It is intended to be a high-level overview of the package's structure and functionality. For a thorough overview, please refer to the :doc:`API reference </api/metasyn>`. Clicking on a class or module name will automatically take you to its API reference section.

.. warning:: 
    This page is intended for developers and advanced users. If you are new to ``metasyn``, please refer to the :doc:`/usage/usage` first. 


MetaFrame class
---------------

The :class:`~metasyn.MetaFrame` class is a core component of the ``metasyn`` package. It represents a metadata frame, which is a structure that holds statistical metadata about a dataset. The data contained in a :obj:`~metasyn.MetaFrame` is in line with the :doc:`/developer/GMF`.

**Attributes:**

- ``meta_vars``: A list of :obj:`~metasyn.MetaVar` objects, each representing a column in a DataFrame.
- ``n_rows``: The number of rows in the original DataFrame.

**Properties:**

- :obj:`~metasyn.MetaFrame.n_columns`: Returns the number of columns in the original DataFrame.
- :obj:`~metasyn.MetaFrame.descriptions`: Returns a dictionary of column descriptions. Can also be set to update the descriptions.

**Methods:**

- :meth:`~metasyn.MetaFrame.fit_dataframe`: Creates a :obj:`~metasyn.MetaFrame` object from a Polars DataFrame. It takes several parameters including the DataFrame, column specifications, distribution providers, privacy level, and a progress bar flag.
- :meth:`~metasyn.MetaFrame.to_dict`: Returns a dictionary with the properties of the :obj:`~metasyn.MetaFrame` for recreation.
- :meth:`~metasyn.MetaFrame.__getitem__`: Returns a :obj:`~metasyn.MetaVar` either by variable name or index.
- :meth:`~metasyn.MetaFrame.__str__`: Returns a formatted string representation of the :obj:`~metasyn.MetaFrame`.
- :meth:`~metasyn.MetaFrame.export`: Serializes and exports the :obj:`~metasyn.MetaFrame` to a JSON file, following the GMF format.
- :meth:`~metasyn.MetaFrame.to_json`: A wrapper for the `export` method.
- :meth:`~metasyn.MetaFrame.from_json`: Reads a :obj:`~metasyn.MetaFrame` from a JSON file.
- :meth:`~metasyn.MetaFrame.synthesize`: Creates a synthetic Polars DataFrame based on the :obj:`~metasyn.MetaFrame`.
- :meth:`~metasyn.MetaFrame.__repr__`: Returns the :obj:`~metasyn.MetaFrame` as it would be output to JSON.


MetaVar class
-------------

The :class:`~metasyn.MetaVar` represents a metadata variable, and is a structure that holds all metadata needed to generate a synthetic column for it. This is the variable level building block for the MetaFrame. It contains the methods to convert a polars `Series` into a variable with an appropriate distribution. The :obj:`~metasyn.MetaVar` class is to the :obj:`~metasyn.MetaFrame` what a polars `Series` is to a `DataFrame`.

**Attributes:**

- ``var_type``: The type of the variable (e.g., continuous, string, etc.).
- ``series``: The (Polars) series from which the variable is created.
- ``name``: The name of the variable/column.
- ``distribution``: The distribution from which random values are drawn.
- ``prop_missing``: The proportion of the series that are missing/NA.
- ``dtype``: The type of the original values (e.g., int64, float, etc.). Used for type-casting back.
- ``description``: A user-provided description of the variable.

**Methods:**

- :meth:`~metasyn.MetaVar.__init__`: Initializes a new instance of the :obj:`~metasyn.MetaVar` class. 
- :meth:`~metasyn.MetaVar.detect`: Detects the variable class(es) of a series or dataframe. This method does not fit any distribution, but it does infer the correct types for the :obj:`~metasyn.MetaVar` and saves the ``Series`` for later fitting.
- :meth:`~metasyn.MetaVar.get_var_type`: Converts a polars ``dtype`` to a ``metasyn`` variable type.
- :meth:`~metasyn.MetaVar.to_dict`: Creates a dictionary from the variable.
- :meth:`~metasyn.MetaVar.__str__`: Returns an easy-to-read formatted string for the variable.
- :meth:`~metasyn.MetaVar.fit`: Fits distributions to the data. Here you can set the distribution, privacy package and uniqueness for the variable again.
- :meth:`~metasyn.MetaVar.draw`: Draws a random item for the variable in whatever type is required.
- :meth:`~metasyn.MetaVar.draw_series`: Draws a new synthetic series from the metadata. For this to work, the variable has to be fitted.
- :meth:`~metasyn.MetaVar.from_dict`: Restores a variable from a dictionary.

Subpackages
-----------
There are currently three subpackages in the ``metasyn`` package. These are the :mod:`~metasyn.distribution`, :mod:`~metasyn.schema`, and :mod:`~metasyn.demo` packages.

Distribution subpackage
^^^^^^^^^^^^^^^^^^^^^^^
The :mod:`~metasyn.distribution` package contains (submodules with) the classes that are used to fit distributions to the data and draw random values from them. More information on distributions and how to implement them can be found in the :doc:`/developer/distributions` documentation page.

Schema subpackage
^^^^^^^^^^^^^^^^^
The :mod:`~metasyn.schema` package simply contains the JSON-schema used to validate metadata, and ensure that it is in line with the :doc:`/developer/GMF`.

Demo subpackage
^^^^^^^^^^^^^^^
The :mod:`~metasyn.demo` package is meant for demo and tutorial purposes. It contains only two functions, :meth:`~metasyn.demo.create_titanic_demo`, which can be used to create a demo dataset based on the `Titanic dataset <https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>`_, and :meth:`~metasyn.demo.demo_file`, which retrieves the filepath to this demo dataset allowing users to quickly access it. 

:meth:`~metasyn.demo.demo_file` is imported automatically as part of the main ``metasyn`` package, as such it can be accessed through :meth:`metasyn.demo_file`, as opposed to :meth:`metasyn.demo.demo_file`. 

Submodules
----------
The ``metasyn`` package is organized into several submodules, each focusing on different aspects of synthetic data generation and privacy. Here's an overview of some key submodules:

var module
^^^^^^^^^^
The :mod:`metasyn.var` module contains the :class:`~metasyn.var.MetaVar` class and its methods, as described above.

metaframe module
^^^^^^^^^^^^^^^^
The :mod:`metasyn.metaframe` module contains the :class:`~metasyn.metaframe.MetaFrame` class and its methods, as described above. 

provider module
^^^^^^^^^^^^^^^
The :mod:`metasyn.provider` module contains the :class:`~metasyn.provider.BaseDistributionProvider` class, which encapsulates a set of distributions, the :class:`~metasyn.provider.BuiltinDistributionProvider` class, which includes the builtin distributions and the :class:`~metasyn.provider.DistributionProviderList` class to allow for multiple distribution providers.

testutils module
^^^^^^^^^^^^^^^^
The :mod:`metasyn.testutils` module provides testing utilities for plugins. It includes functions for checking distributions and distribution providers.

validation module
^^^^^^^^^^^^^^^^^
The :mod:`metasyn.validation` module contains tools for validating distribution outputs and GMF file formats.

privacy module
^^^^^^^^^^^^^^
The :mod:`metasyn.privacy` module contains the basis for implementing privacy features.

A system to incorporate privacy features such as differential privacy or other forms of disclosure control is still being implemented.

