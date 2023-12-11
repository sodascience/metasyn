Distributions 
=============

Distributions subpackage
------------------------

The :mod:`~metasyn.distribution` is an essential subpackage that contains (submodules with) classes used to represent and handle various distributions. 

The modules are :mod:`~metasyn.distribution.base`, :mod:`~metasyn.distribution.categorical`, :mod:`~metasyn.distribution.continuous`, :mod:`~metasyn.distribution.datetime`, :mod:`~metasyn.distribution.discrete`, :mod:`~metasyn.distribution.faker`, :mod:`~metasyn.distribution.na` and :mod:`~metasyn.distribution.regex`. 

:mod:`~metasyn.distribution.base` contains the base classes for all distributions, while the other modules implement distributions for their respective variable types.

An overview of the distributions each module implements, and from which it inherits, is shown below:

.. image:: /images/distributions.svg
   :alt: Metasyn distributions
   :scale: 100%

Base submodule
~~~~~~~~~~~~~~
The base module contains the :class:`~metasyn.distribution.BaseDistribution` class, which is the base class for all distributions. It also contains the :class:`~metasyn.distribution.ScipyDistribution` class, which is a specialized base class for distributions that are built on top of SciPy's statistical distributions. 

Additionally it contains the :class:`~metasyn.distribution.UniqueDistributionMixin` class, which is a mixin class that can be used to indicate that a distribution is unique (i.e., that it does not contain duplicate values).

Finally it contains the :func:`~metasyn.distribution.metadist` decorator, which is used to set the attributes of a distribution (e.g., ``implements``, ``var_type``, etc.).

BaseDistribution class
^^^^^^^^^^^^^^^^^^^^^^
This is the base class providing the basic structure for all distributions. It is not intended to be used directly, but rather to be derived from when implementing a new distribution.

**Attributes:**

- ``implements``: A unique string identifier for the distribution type, e.g. ``core.uniform`` or ``core.poisson``.
- ``var_type``: The type of variable associated with the distribution, e.g. ``discrete`` or ``continuous``.
- ``provenance``: Information about the source (core, plugin, etc.) of the distribution.
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
- :meth:`~metasyn.distribution.BaseDistribution.draw_reset`: Method to reset the distribution's drawing mechanism. This should be implemented if the subsequent draws are not independent.
- :meth:`~metasyn.distribution.BaseDistribution._param_dict`: Abstract method to return a dictionary of the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution.to_dict`: Method to create a dictionary representation of the distribution. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.from_dict`: Class method to create a distribution from a dictionary. 
- :meth:`~metasyn.distribution.BaseDistribution._param_schema`: Abstract method intended to return a schema for the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution.schema`: Class method to generate a JSON schema to validate the distribution's structure.
- :meth:`~metasyn.distribution.BaseDistribution.information_criterion`: Class method to determine which distribution gets selected during the fitting process for a series of values. The distribution with the lowest information criterion with the correct variable type will be selected. For discrete and continuous distributions it is currently implemented as `BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_). It is recommended to be implemented by derived classes.
- :meth:`~metasyn.distribution.BaseDistribution.matches_name`: Class method to check if a distribution matches a given name (specified in the ``implements`` field).
- :meth:`~metasyn.distribution.BaseDistribution.default_distribution`: Abstract class method
- ``__str__``: Overridden method to return a formatted string representation of the distribution.

.. warning:: 
  When implementing a new distribution, the :meth:`~metasyn.distribution.BaseDistribution._fit`, :meth:`~metasyn.distribution.BaseDistribution.draw`, and :meth:`~metasyn.distribution.BaseDistribution.to_dict` methods *must* be implemented. 

ScipyDistribution class
^^^^^^^^^^^^^^^^^^^^^^^
The :class:`~metasyn.distribution.ScipyDistribution` is a specialized base class for distributions that are based on
`SciPy <https://docs.scipy.org/doc/scipy/index.html>`_ statistical distributions. 

All the :mod:`~metasyn.distribution.datetime`, :mod:`~metasyn.distribution.discrete`, and :mod:`~metasyn.distribution.continuous` distributions are derived from this class.


UniqueDistributionMixin class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`~metasyn.distribution.UniqueDistributionMixin` is a mixin class that can be combined with other distribution classes to create distributions that generate unique values.

For example, the unique variants of the :class:`metasyn.distribution.regex.RegexDistribution` and the :class:`metasyn.distribution.faker.UniqueFakerDistribution` are implemented using this mixin as follows:

.. code-block:: python

    @metadist(implements="core.unique_regex", var_type="string", is_unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python

    @metadist(implements="core.unique_faker", var_type="string")
    class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):


Metadist decorator method
^^^^^^^^^^^^^^^^^^^^^^^^^
When implementing a new distribution, the ``metadist`` decorator helps set the class attributes of that distribution (e.g. ``implements``, ``var_type``, etc.). 

**Parameters:**

- ``implements``: A unique name for the distribution type, e.g. ``core.uniform`` or ``core.poisson``. 
- ``var_type``: The type of variable associated with the distribution, e.g. ``discrete`` or ``continuous``.
- ``provenance``: Information about the source (core, plugin, etc.) of the distribution.
- ``privacy``: The privacy class or implementation associated with the distribution.
- ``is_unique``: A boolean indicating whether the values in the distribution are unique.
- ``version``: The version of the distribution. 

.. admonition:: Note

    Note that the parameters are the same as the attributes of the :class:`~metasyn.distribution.BaseDistribution` class.

To use the ``metadist`` decorator, annotate the custom distribution class with ``@metadist``, passing in the attributes of the target distribution as parameters.

For example, the following distributions use the decorator as follows:

.. code-block:: python

    @metadist(implements="core.multinoulli", var_type=["categorical", "discrete", "string"])
    class MultinoulliDistribution(BaseDistribution):

.. code-block:: python

    @metadist(implements="core.unique_regex", var_type="string", is_unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python
      
    @metadist(implements="core.uniform_date", var_type="date")
    class UniformDateDistribution(BaseUniformDistribution):


The ``metadist`` decorator is implemented automatically as part of the main ``metasyn`` package. 


Categorical submodule
~~~~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.categorical` module contains the :class:`metasyn.distribution.categorical.MultinoulliDistribution` class, which is used for categorical distributions.

Continuous submodule
~~~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.continuous` module contains the classes used for continuous distributions.

DateTime submodule
~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.datetime` module contains the classes used for ``DateTime`` distributions.

Discrete submodule
~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.discrete` module contains the classes used to for discrete distributions.

Faker submodule
~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.faker` module contains the classes used for distributions that are based on the `Faker <https://faker.readthedocs.io/en/master/>`_ package.

NA submodule
~~~~~~~~~~~~
The :mod:`~metasyn.distribution.na` module contains the :class:`metasyn.distribution.NADistribution` class, a distribution which generates *only* NA values.

Regex submodule
~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.regex` module contains the classes for distributions that are based on regular expressions. It implements the open source `regexmodel <https://github.com/sodascience/regexmodel>`_ package. 

There is also a legacy module :mod:`~metasyn.distribution.legacy.regex` that contains the old implementation of the regex distribution. 


Creating a new distribution
---------------------------

New distributions can be created by either directly inheriting from a base class such as the :class:`~metasyn.distribution.BaseDistribution` (or a specialized base class such as :class:`~metasyn.distribution.ScipyDistribution`)

For example, let's say we want to implement a new series of similar distributions, we can create a base class inheriting from :class:`~metasyn.distribution.BaseDistribution`:

.. code-block:: python

    from metasyn.distribution.base import BaseDistribution, metadist


    class BaseNewDistribution(BaseDistribution):
        """Base class for new distribution.
        This base class makes it easy to implement variations of this distribution for different variable types.
        """

        # implementation details here

Then we can create distributions for different variable types by inheriting from this base class, and using the ``metadist`` decorator to set the proper attributes. 

.. admonition:: Important

    In cases where there are multiple variations of a distribution for different data types, such as in the example given in this section, or the ``core.uniform`` distributions in ``metasyn``. 
    
    In this case, the metasyn convention is to put the data type **after** the distribution name, e.g. ``core.uniform_discrete``, ``core.uniform_datetime``, ``core.uniform_date``, etc.

We can create a new distribution for continuous, discrete and string variables as follows (note that the implementation here is a basic implementation serving as an example):

.. code-block:: python

    @metadist(implements="core.newdistribution", var_type="continuous")
    class NewContinuousDistribution(BaseNewDistribution):
        """Variant of the new distribution for continuous vars."""

        @classmethod
        def default_distribution(cls) -> BaseDistribution:
            return cls(0.0)

        @classmethod
        def _param_schema(cls):
            return {
                "value": {"type": "number"}
            }


    @metadist(implements="core.newdistribution_discrete", var_type="discrete")
    class DiscreteNewDistribution(BaseNewDistribution):
        """Variant of new distribution for discrete vars."""

        @classmethod
        def default_distribution(cls) -> BaseDistribution:
            return cls(0)

        @classmethod
        def _param_schema(cls):
            return {
                "value": {"type": "integer"}
            }

    @metadist(implements="core.newdistribution_string", var_type="string")
    class StringNewDistribution(NewDistribution):
        """Variant of new distribution for string vars."""

        @classmethod
        def default_distribution(cls) -> BaseDistribution:
            return cls("text")

        @classmethod
        def _param_schema(cls):
            return {
                "value": {"type": "string"}
            }