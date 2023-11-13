Structural Overview
===============

This page features a detailed overview of metasyn's modules, with the goal of helping developers get started developing for metasyn.

.. warning:: 
  This page is unfinished and might be outdated, if information is lacking or does not seem right, feel free to :doc:`get in touch </about/contact>`  and we'll try to help you.



Generative Metadata Format
--------------------------

The core of the metasyn is being able to generate statistical metadata for any tabular dataset, and being able to generate synthetic data based on this. This is a file in with the JSON format, that describes a *dataset*.
It includes on the base level:

* ``n_rows``: Number of rows
* ``n_columns``: Number of columns
* ``provenance``: Details on how the file was created.
* ``vars``: Description of all variables (columns) of the dataset.

Each *variable* in the list has at least the following attributes:

* ``name``: Name of the variable (column name).
* ``type``: Type of variable
* ``dtype``: This is the polars dtype that the variable has to be converted back to.
* ``prop_missing``: The proportion of values that are NA.
* ``distribution``: The distribution that describes the variable.

Each *distribution* has at least two attributes:

* ``name``: Name of the distribution.
* ``parameters``: Dictionary containing the parameters of the distribution.

The full schema that describes the structure of this file is included in the
`GitHub repository <https://github.com/sodascience/meta-synth/blob/main/metasyn/schema/metasyn-1_0.json>`_.



Module Architecture
-------------------


MetaFrame
~~~~~~~~~

The :class:`metasyn.MetaFrame` class is a core component of the ``metasyn`` package. It represents a metadata frame, which is a structure that holds metadata about a dataset. 

**Fields:**

- ``meta_vars``: A list of :obj:`metasyn.MetaVar` objects, each representing a column in a DataFrame.
- ``n_rows``: The number of rows in the original DataFrame.

**Properties:**

- :obj:`metasyn.MetaFrame.n_columns`: Returns the number of columns in the original DataFrame.
- :obj:`metasyn.MetaFrame.descriptions`: Returns a dictionary of column descriptions. Can also be set to update the descriptions.

**Methods:**

- :meth:`metasyn.MetaFrame.fit_dataframe`: Creates a `MetaFrame` object from a Polars DataFrame. It takes several parameters including the DataFrame, column specifications, distribution providers, privacy level, and a progress bar flag.
- :meth:`metasyn.MetaFrame.to_dict`: Returns a dictionary with the properties of the `MetaFrame` for recreation.
- :meth:`metasyn.MetaFrame.__getitem__`: Returns a `MetaVar` either by variable name or index.
- :meth:`metasyn.MetaFrame.__str__`: Returns a formatted string representation of the `MetaFrame`.
- :meth:`metasyn.MetaFrame.export`: Serializes and exports the `MetaFrame` to a JSON file, following the GMF format.
- :meth:`metasyn.MetaFrame.to_json`: A wrapper for the `export` method.
- :meth:`metasyn.MetaFrame.from_json`: Reads a `MetaFrame` from a JSON file.
- :meth:`metasyn.MetaFrame.synthesize`: Creates a synthetic Polars DataFrame based on the `MetaFrame`.
- :meth:`metasyn.MetaFrame.__repr__`: Returns the `MetaFrame` as it would be output to JSON.

**Relation to other classes**

- :obj:`metasyn.MetaVar`: A `MetaFrame` is composed of a list of :obj:`metasyn.MetaVar` objects, each of which represents a column in the DataFrame. 
- :mod:`metasyn.privacy.BasePrivacy` and :mod:`metasyn.privacy.BasicPrivacy```: These are used to set the privacy level when creating a `MetaFrame` from a DataFrame.
- :mod:`metasyn.provider.BaseDistributionProvider`: This is used to set the distribution providers when creating a `MetaFrame` from a DataFrame.

MetaVar
~~~~~~~

The :class:`metasyn.MetaVar` represents a metadata variable, and is a structure that holds all metadata needed to generate a synthetic column for it. This is the variable level building block for the MetaFrame. It contains the methods to convert a polars `Series` into a variable with an appropriate distribution. The `MetaVar` class is to the `MetaFrame` what a polars `Series` is to a `DataFrame`.

**Fields:**

- ``var_type``: The type of the variable (e.g., continuous, string, etc.).
- ``series``: The (Polars) series from which the variable is created.
- ``name``: The name of the variable/column.
- ``distribution``: The distribution from which random values are drawn.
- ``prop_missing``: The proportion of the series that are missing/NA.
- ``dtype``: The type of the original values (e.g., int64, float, etc.). Used for type-casting back.
- ``description``: A user-provided description of the variable.

**Methods:**

- :meth:`metasyn.MetaVar.__init__`: Initializes a new instance of the ``MetaVar`` class. 
- :meth:`metasyn.MetaVar.detect`: Detects the variable class(es) of a series or dataframe. This method does not fit any distribution, but it does infer the correct types for the ``MetaVar`` and saves the ``Series`` for later fitting.
- :meth:`metasyn.MetaVar.get_var_type`: Converts a polars ``dtype`` to a metasyn variable type.
- :meth:`metasyn.MetaVar.to_dict`: Creates a dictionary from the variable.
- :meth:`metasyn.MetaVar.__str__`: Returns an easy-to-read formatted string for the variable.
- :meth:`metasyn.MetaVar.fit`: Fits distributions to the data. Here you can set the distribution, privacy package and uniqueness for the variable again.
- :meth:`metasyn.MetaVar.draw`: Draws a random item for the variable in whatever type is required.
- :meth:`metasyn.MetaVar.draw_series`: Draws a new synthetic series from the metadata. For this to work, the variable has to be fitted.
- :meth:`metasyn.MetaVar.from_dict`: Restores a variable from a dictionary.


**Relation to other classes**

- :mod:`metasyn.distribution.BaseDistribution`: Used to represent the distribution of the variable.
- :mod:`metasyn.privacy.BasePrivacy`: Represents the privacy level used for fitting the series.
- :mod:`metasyn.provider.BaseDistributionProvider`: Provides distributions for fitting.




MetaDistribution
~~~~~~~~~~~~~~~~

This class will likely be removed at some point due to a lack of functionality. Only the fit function is currently used by
`MetaVar`. It should be either be removed or added to a new `DistributionTree` class.

Distributions
-------------

All distribution code can be found in `metasyn.distribution`. The distributions are somewhat ordered by variable type.

BaseDistribution
~~~~~~~~~~~~~~~~

This distribution is the basis of all distributions, and every distribution that is defined indirectly derives from this base.
It is mostly a collection of abstract methods that have to be implemented by any derived class. The following are
recommended/mandatory for derived classes to implement:

* :meth:`_fit <metasyn.distribution.base.BaseDistribution._fit>`:
  This method should fit the distribution to the values. (**mandatory**)
* :meth:`draw <metasyn.distribution.base.BaseDistribution.draw>`:
  This method should draw a new value from the distribution. (**mandatory**)
* :meth:`draw_reset <metasyn.distribution.base.BaseDistribution.draw_reset>`:
  This method needs to be set if subsequent values are not independent (**optional**).
* :meth:`to_dict <metasyn.distribution.base.BaseDistribution.to_dict>`: 
  This method creates a JSON compatible representation of the distribution (**mandatory**).
* :meth:`information_cirterion <metasyn.distribution.base.BaseDistribution.information_criterion>`:
  This method determines the relative priority of the
  distributions. Currently implemented as Akaike Information Criterion for the discrete and continuous distributions.
  (**recommended**)
* :meth:`fit_kwargs <metasyn.distribution.base.BaseDistribution.fit_kwargs>`:
  This is currently only implemented for the faker distribution, and
  it allows us to use `faker.city` as a distribution. (**optional**)
* :meth:`_example_distribution <metasyn.distribution.base.BaseDistribution._example_distribution>`:
  Return a distribution with some parameters to test.
  This is currently mainly used for the continuous integration/testing. (**mandatory**)

Apart from the methods to be implemented, there are also attributes that should be set:

* ``aliases``: A list of aliases/names for the distribution. The first alias should be the name of the distribution as it is
  presented in the Metasyn File. (**mandatory**)
* ``is_unique``: Set to true if the distribution always generates unique outputs. (**optional**)

Variable type specific distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each variable type a class is derived from the ``BaseDistribution``. It sets the ``var_type`` which is used in the ``MetaVar``
class and the Metasyn File. A distribution should always derive from one of those distributions, either directly or indirectly.

ScipyDistribution
~~~~~~~~~~~~~~~~~

This distribution is useful for discrete and continuous distributions that are based on
`SciPy <https://docs.scipy.org/doc/scipy/index.html>`_. Most of the currently implemented numerical distributions
use the ``ScipyDistribution`` as their base class (while also having either ``DiscreteDistribution`` or ``ContinuousDistribution``
as a baseclass).

:mod:`Privacy Features (experimental) <metasyn.privacy>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A system to incorporate privacy features such as differential privacy or other forms of disclosure control is being implemented.
This part of the code is considered to be particularly unstable, so modifications for future versions are likely necessary.

The :mod:`metasyn.privacy.cbs` sub-package is an example of how to implement a privacy package. Notice that all distributions
are derived from their non-private counterparts in :mod:`metasyn.distribution`. Only distributions that are derived in the
privacy package are available while fitting. Thus, if the privacy package simply wants the copy the distribution from the main
package it should simply use class derivation and add a docstring, such as :class:`metasyn.privacy.cbs.continuous.CbsNormal`.

The :mod:`metasyn.privacy.cbs` sub-package will be removed at some point and possibly be redistributed as its own package if
there is demand for it.



