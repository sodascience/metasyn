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

The :class:`~metasyn.MetaFrame` class is a core component of the ``metasyn`` package. It represents a metadata frame, which is a structure that holds metadata about a dataset. 

**Fields:**

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

.. **Relation to other classes**

.. - :obj:`~metasyn.MetaVar`: A :obj:`~metasyn.MetaFrame` is composed of a list of :obj:`~metasyn.MetaVar` objects, each of which represents a column in the DataFrame. 
.. - :mod:`~metasyn.privacy.BasePrivacy` and :mod:`~metasyn.privacy.BasicPrivacy`: These are used to set the privacy level when creating a :obj:`~metasyn.MetaFrame` from a DataFrame.
.. - :mod:`~metasyn.provider.BaseDistributionProvider`: This module is used to set the distribution providers when creating a :obj:`~metasyn.MetaFrame` from a DataFrame.

MetaVar
~~~~~~~

The :class:`~metasyn.MetaVar` represents a metadata variable, and is a structure that holds all metadata needed to generate a synthetic column for it. This is the variable level building block for the MetaFrame. It contains the methods to convert a polars `Series` into a variable with an appropriate distribution. The :obj:`~metasyn.MetaVar` class is to the :obj:`~metasyn.MetaFrame` what a polars `Series` is to a `DataFrame`.

**Fields:**

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
- :meth:`~metasyn.MetaVar.get_var_type`: Converts a polars ``dtype`` to a metasyn variable type.
- :meth:`~metasyn.MetaVar.to_dict`: Creates a dictionary from the variable.
- :meth:`~metasyn.MetaVar.__str__`: Returns an easy-to-read formatted string for the variable.
- :meth:`~metasyn.MetaVar.fit`: Fits distributions to the data. Here you can set the distribution, privacy package and uniqueness for the variable again.
- :meth:`~metasyn.MetaVar.draw`: Draws a random item for the variable in whatever type is required.
- :meth:`~metasyn.MetaVar.draw_series`: Draws a new synthetic series from the metadata. For this to work, the variable has to be fitted.
- :meth:`~metasyn.MetaVar.from_dict`: Restores a variable from a dictionary.


.. **Relation to other classes**

.. - :mod:`~metasyn.distribution.BaseDistribution`: This is the base class for all distributions. It is used to set the distribution when fitting a variable.
.. - :mod:`~metasyn.privacy.BasePrivacy`: Represents the privacy level used for fitting the series.
.. - :mod:`~metasyn.provider.BaseDistributionProvider`: This module is used to set the pool of distributions from which to choose when fitting a variable.


Distributions
-------------

The :mod:`~metasyn.distribution` module contains the classes used to represent distributions. It is composed of a base class (:class:`~metasyn.distribution.BaseDistribution`) and several derived classes, each representing a different type of distribution. The base class provides the basic structure for all distributions, and is not intended to be used directly. Rather, it is intended to be derived from when implementing a new distribution.

An overview of all the (built-in) distributions can be seen below.

.. image:: /images/distributions.svg
   :alt: Metasyn distributions
   :scale: 100%


BaseDistribution
~~~~~~~~~~~~~~~~

This is the base class providing the basic structure for all distributions. It is not intended to be used directly, but rather to be derived from when implementing a new distribution.

**Fields:**

- ``implements``: A unique string identifier for the distribution type, e.g. ``core.discrete_uniform`` or ``core.poisson``.
- ``var_type``: The type of variable associated with the distribution, e.g. ``discrete`` or ``continuous``.
- ``provenance``: Information about the source of the distribution.
- ``privacy``: The privacy class or implementation associated with the distribution.
- ``is_unique``: A boolean indicating whether the values in the distribution are unique.
- ``version``: The version of the distribution.

**Properties:**

- :meth:`~metasyn.distribution.BaseDistribution._params_formatted`: Provides a formatted string of the distribution's parameters for easy readability.

**Methods:**

- :meth:`~metasyn.distribution.BaseDistribution.fit`: Class method to fit a distribution to a given series. 
- :meth:`~metasyn.distribution.BaseDistribution._fit`: Abstract class method intended to contain the fitting logic for the distribution. It does not need to handle N/A values. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution._to_series`: Static method converting different data types (Polars Series, Pandas Series, or sequences) into a Polars Series, handling null values appropriately.
- :meth:`~metasyn.distribution.BaseDistribution.draw`: Abstract method, intended to draw a new value from the distribution. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.draw_reset`: Method to reset the distribution's drawing mechanism. This should be implemented if the drawing does not happen randomly.
- :meth:`~metasyn.distribution.BaseDistribution._param_dict`: Abstract method to return a dictionary of the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution.to_dict`: Method to create a dictionary representation of the distribution. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.from_dict`: Class method to create a distribution from a dictionary. 
- :meth:`~metasyn.distribution.BaseDistribution._param_schema`: Abstract method intended to return a schema for the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution.schema`: Class method to generate a JSON schema to validate the distribution's structure.
- :meth:`~metasyn.distribution.BaseDistribution.information_criterion`: Class method to determine the relative priority (information criterion) for a series of values. For discrete and continuous distributions it is currently implemented as `BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_). It is recommended to be implemented by derived classes.
- :meth:`~metasyn.distribution.BaseDistribution.matches_name`: Class method to check if a distribution matches a given name (specified in the ``implements`` field).
- :meth:`~metasyn.distribution.BaseDistribution.default_distribution`: Abstract class method
- ``__str__``: Overridden method to return a formatted string representation of the distribution.

.. warning:: 
  When implementing a new distribution, the :meth:`~metasyn.distribution.BaseDistribution._fit`, :meth:`~metasyn.distribution.BaseDistribution.draw`, and :meth:`~metasyn.distribution.BaseDistribution.to_dict` methods must be implemented. 


metadist decorator
^^^^^^^^^^^^^^^^^^
When implementing a new distribution, you can use the ``metadist`` decorator to specify its attributes.


**Usage:**

To use the ``metadist`` decorator, annotate the custom distribution class with ``@metadist`` and provide relevant parameters to define specific characteristics of the distribution.

**Parameters:**

- ``implements``: A unique string identifier for the distribution type, such as ``core.discrete_uniform`` or ``core.poisson``.
- ``var_type``: Specifies the variable type associated with the distribution, e.g., ``discrete`` or ``continuous``.
- ``is_unique``: (Optional) A boolean flag indicating if the distribution produces unique values.
- ``privacy``: (Optional) Specifies the privacy class or implementation.
- ``version``: (Optional) The version of the distribution, useful for tracking updates and compatibility.

**Examples:**

.. code-block:: python

   @metadist(implements="core.discrete_uniform", var_type="discrete")
   class DiscreteUniformDistribution(ScipyDistribution):
       # Implementation details here

   @metadist(implements="core.poisson", var_type="discrete")
   class PoissonDistribution(ScipyDistribution):
       # Implementation details here

   @metadist(implements="core.unique_key", var_type="discrete", is_unique=True)
   class UniqueKeyDistribution(ScipyDistribution):
       # Implementation details here




.. Variable type specific distributions
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. For each variable type a class is derived from the ``BaseDistribution``. It sets the ``var_type`` which is used in the :obj:`~metasyn.MetaVar``
.. class and the Metasyn File. A distribution should always derive from one of those distributions, either directly or indirectly.

.. ScipyDistribution
.. ~~~~~~~~~~~~~~~~~

.. This distribution is useful for discrete and continuous distributions that are based on
.. `SciPy <https://docs.scipy.org/doc/scipy/index.html>`_. Most of the currently implemented numerical distributions
.. use the ``ScipyDistribution`` as their base class (while also having either ``DiscreteDistribution`` or ``ContinuousDistribution``
.. as a baseclass).

.. :mod:`~Privacy Features (experimental) <metasyn.privacy>`
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. A system to incorporate privacy features such as differential privacy or other forms of disclosure control is being implemented.
.. This part of the code is considered to be particularly unstable, so modifications for future versions are likely necessary.

.. The :mod:`~metasyn.privacy.cbs` sub-package is an example of how to implement a privacy package. Notice that all distributions
.. are derived from their non-private counterparts in :mod:`~metasyn.distribution`. Only distributions that are derived in the
.. privacy package are available while fitting. Thus, if the privacy package simply wants the copy the distribution from the main
.. package it should simply use class derivation and add a docstring, such as :class:`~metasyn.privacy.cbs.continuous.CbsNormal`.

.. The :mod:`~metasyn.privacy.cbs` sub-package will be removed at some point and possibly be redistributed as its own package if
.. there is demand for it.



