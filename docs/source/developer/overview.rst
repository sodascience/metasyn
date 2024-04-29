Overview
========

This page aims to introduce the main classes and modules of the ``metasyn`` package. It is intended to be a high-level overview of the package's structure and functionality. For a thorough overview, please refer to the :doc:`API reference </api/metasyn>`. Clicking on a class or module name will automatically take you to its API reference section.

.. warning:: 
    This page is intended for developers and advanced users. If you are new to ``metasyn``, please refer to the :doc:`/usage/usage` first. 


MetaFrame class
---------------

The :class:`~metasyn.MetaFrame` class is a core component of the ``metasyn`` package. It represents a metadata frame, which is a structure that holds statistical metadata about a dataset. The data contained in a :obj:`~metasyn.MetaFrame` is in line with the :doc:`/developer/GMF`.

Essentially, a :obj:`~metasyn.MetaFrame` is a collection of :obj:`~metasyn.MetaVar` objects, each representing a column in a dataset. It contains methods that allow for the following:

- **Fitting to a DataFrame**: The :meth:`~metasyn.MetaFrame.fit_dataframe` method allows for fitting a Polars DataFrame to create a :obj:`~metasyn.MetaFrame` object. This method takes several parameters including the DataFrame, column specifications, distribution providers, privacy level, and a progress bar flag.
- **Exporting and importing**: The :meth:`~metasyn.MetaFrame.export` method serializes and exports the :obj:`~metasyn.MetaFrame` to a JSON file, following the GMF format. The :meth:`~metasyn.MetaFrame.from_json` method reads a :obj:`~metasyn.MetaFrame` from a JSON file.
- **Synthesizing to a DataFrame**: The :meth:`~metasyn.MetaFrame.synthesize` method creates a synthetic Polars DataFrame based on the :obj:`~metasyn.MetaFrame`.


MetaVar class
-------------

The :class:`~metasyn.MetaVar` represents a metadata variable, and is a structure that holds all metadata needed to generate a synthetic column for it. This is the variable level building block for the MetaFrame. It contains the methods to convert a polars `Series` into a variable with an appropriate distribution. The :obj:`~metasyn.MetaVar` class is to the :obj:`~metasyn.MetaFrame` what a polars `Series` is to a `DataFrame`.

A :obj:`~metasyn.MetaVar` contains information on the variable type (``var_type``), the series from which the variable is created (``series``), the name of the variable (``name``), the distribution from which random values are drawn (``distribution``), the proportion of the series that are missing/NA (``prop_missing``), the type of the original values (``dtype``), and a user-provided description of the variable (``description``). 

This class is considered a passthrough class used by the :obj:`~metasyn.MetaFrame` class, and is not intended to be used directly by the user. It contains the following functionality:

- **Fitting distributions**: The :meth:`~metasyn.MetaVar.fit` method fits distributions to the data. Here you can set the distribution, privacy package and uniqueness for the variable.
- **Drawing values and series**: The :meth:`~metasyn.MetaVar.draw` method draws a random item for the variable in whatever type is required. The :meth:`~metasyn.MetaVar.draw_series` method draws a new synthetic series from the metadata. For this to work, the variable has to be fitted.
- **Converting to and from a dictionary**: The :meth:`~metasyn.MetaVar.to_dict` method creates a dictionary from the variable. The :meth:`~metasyn.MetaVar.from_dict` method restores a variable from a dictionary.


Subpackages
-----------
There are currently three subpackages in the ``metasyn`` package. These are the :mod:`~metasyn.distribution`, :mod:`~metasyn.schema`, and :mod:`~metasyn.demo` packages.

* the :mod:`~metasyn.distribution` subpackage contains (submodules with) the classes that are used to fit distributions to the data and draw random values from them. More information on distributions and how to implement them can be found in the :doc:`/developer/distributions` documentation page.
* The :mod:`~metasyn.schema` package simply contains the JSON-schema used to validate metadata, and ensure that it is in line with the :doc:`/developer/GMF`.
* The :mod:`~metasyn.demo` package is meant for demo and tutorial purposes. It contains only two functions, :meth:`~metasyn.demo.create_titanic_demo`, which can be used to create a demo dataset based on the `Titanic dataset <https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>`_, and :meth:`~metasyn.demo.demo_file`, which retrieves the filepath to this demo dataset allowing users to quickly access it. 

:meth:`~metasyn.demo.demo_file` is imported automatically as part of the main ``metasyn`` package, as such it can be accessed through :meth:`metasyn.demo_file`, as opposed to :meth:`metasyn.demo.demo_file`. 

Submodules
----------
A comprehensive overview of metasyn and all its modules can be found in the API reference's :doc:`/api/developer_reference` documentation page.