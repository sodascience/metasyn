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

Additionally it contains the :class:`~metasyn.distribution.UniqueDistributionMixin` class, which is a mixin class that can be used to make a distribution unique (i.e., one that does not contain duplicate values).

Finally it contains the :func:`~metasyn.distribution.metadist` decorator, which is used to set the attributes of a distribution (e.g., ``implements``, ``var_type``, etc.).

BaseDistribution class
^^^^^^^^^^^^^^^^^^^^^^
This is the base class providing the basic structure for all distributions. It is not intended to be used directly, but rather to be derived from when implementing a new distribution.

**Attributes:**

- ``implements``: A unique string identifier for the distribution type, e.g. ``core.uniform`` or ``user.months``. The naming convention for the ``implements`` attribute is ``<provider_name>.<distribution_name>``. Distributions that are part of the core metasyn distribution provider should use ``core`` as the provider name.
- ``var_type``: The type of variable associated with the distribution, e.g. ``discrete`` or ``continuous``.
- ``provenance``: Information about the source (core, plugin, etc.) of the distribution.
- ``privacy``: The privacy class or implementation associated with the distribution.
- ``is_unique``: A boolean indicating whether the values in the distribution are unique.
- ``version``: The version of the distribution. 

Though they can be set manually, the intended way of setting these attributes is through the use of the ``metadist`` decorator (which is covered in the next section).

**Properties:**

- :meth:`~metasyn.distribution.BaseDistribution._params_formatted`: Provides a formatted string of the distribution's parameters for easy readability.

**Methods:**

- :meth:`~metasyn.distribution.BaseDistribution.fit`: Class method to fit a distribution to a given series. 
- :meth:`~metasyn.distribution.BaseDistribution._fit`: Abstract class method intended to contain the fitting logic for the distribution. It does not need to handle N/A values. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution._to_series`: Static method converting different data types (Polars Series, Pandas Series, or sequences) into a Polars Series, handling null values appropriately.
- :meth:`~metasyn.distribution.BaseDistribution.draw`: Abstract method, intended to draw a new value from the distribution. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.draw_reset`: Method to reset the distribution's drawing mechanism. This should be implemented if the subsequent draws are not independent.
- :meth:`~metasyn.distribution.BaseDistribution._param_dict`: Abstract method to return a dictionary of the distribution's parameters. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.to_dict`: Method to create a dictionary representation of the distribution. 
- :meth:`~metasyn.distribution.BaseDistribution.from_dict`: Class method to create a distribution from a dictionary. 
- :meth:`~metasyn.distribution.BaseDistribution._param_schema`: Abstract method intended to return a schema for the distribution's parameters. **It must be implemented by derived classes.**
- :meth:`~metasyn.distribution.BaseDistribution.schema`: Class method to generate a JSON schema to validate the distribution's structure.
- :meth:`~metasyn.distribution.BaseDistribution.information_criterion`: Class method to determine which distribution gets selected during the fitting process for a series of values. The distribution with the lowest information criterion with the correct variable type will be selected. For discrete and continuous distributions it is currently implemented as `BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_). It is recommended to be implemented by derived classes.
- :meth:`~metasyn.distribution.BaseDistribution.matches_name`: Class method to check if a distribution matches a given name (specified in the ``implements`` field).
- :meth:`~metasyn.distribution.BaseDistribution.default_distribution`: Abstract class method intended to return a distribution with default parameters. **It must be implemented by derived classes.**
- ``__str__``: Overridden method to return a formatted string representation of the distribution. 

.. warning:: 
  When implementing a new distribution, the :meth:`~metasyn.distribution.BaseDistribution._fit`, :meth:`~metasyn.distribution.BaseDistribution.draw`, :meth:`~metasyn.distribution.BaseDistribution._param_dict`, :meth:`~metasyn.distribution.BaseDistribution._param_schema` and :meth:`~metasyn.distribution.BaseDistribution.default_distribution` methods *must* be implemented. 

ScipyDistribution class
^^^^^^^^^^^^^^^^^^^^^^^
The :class:`~metasyn.distribution.ScipyDistribution` is a specialized base class for distributions that are based on
`SciPy <https://docs.scipy.org/doc/scipy/index.html>`_ statistical distributions. 

All the current :mod:`~metasyn.distribution.discrete` and :mod:`~metasyn.distribution.continuous` distributions are derived from this class.


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

.. warning:: 
    
    This mixin class has a default implementation that will work for many distributions, but it may not be appropriate for all. Be sure to check the implementation before using it. 

Metadist decorator method
^^^^^^^^^^^^^^^^^^^^^^^^^
When implementing a new distribution, the ``metadist`` decorator is intended to be used to set the class attributes of that distribution (e.g. ``implements``, ``var_type``, etc.). Refer to :class:`~metasyn.distribution.BaseDistribution` for an overview of these attributes.

To use the ``metadist`` decorator, annotate a distribution class with ``@metadist``, passing in the attributes of the target distribution as parameters.

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


The ``metadist`` decorator, which is a part of the :mod:`metasyn.distribution.base` submodule, is directly accessible when importing the main metasyn package, as it's explicitly and relatively imported upon importing the main metasyn package.

Categorical submodule
~~~~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.categorical` module contains the :class:`metasyn.distribution.categorical.MultinoulliDistribution` class, which is used for categorical distributions.

Continuous submodule
~~~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.continuous` module contains the classes used for continuous distributions.

DateTime submodule
~~~~~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.datetime` module contains the classes used for the ``time``, ``date`` and ``datetime`` distributions.

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

Creating a new distribution
---------------------------
The first step to creating a new distribution is to inherit from a distribution class. This can be a base class (e.g. :class:`~metasyn.distribution.BaseDistribution`, :class:`~metasyn.distribution.ScipyDistribution`), or an existing distribution. It is also possible to create a new base class and inherit from there, though this is not recommended.

The next step is to set the attributes of the distribution using the ``metadist`` decorator. Refer to :class:`~metasyn.distribution.BaseDistribution` for an overview of these attributes.

.. admonition:: Important

    In cases where there are multiple variations of a distribution for different data types, for example as is the case with the ``core.uniform`` distributions in ``metasyn``, the convention is to put the data type **after** the distribution name. For example: ``core.uniform_discrete``, ``core.uniform_datetime``, ``core.uniform_date``, etc.


Then, implement the required methods (:meth:`~metasyn.distribution.BaseDistribution._fit`, :meth:`~metasyn.distribution.BaseDistribution.draw`, :meth:`~metasyn.distribution.BaseDistribution._param_dict`, :meth:`~metasyn.distribution.BaseDistribution._param_schema` and :meth:`~metasyn.distribution.BaseDistribution.default_distribution`), as well as any other applicable methods.

Finally the distribution has to be added to a provider list, so that it can be used by ``metasyn`` for fitting.

For example, let's say we want to create a new distribution for unique continuous variables, to be a part of the core ``metasyn`` distribution provider. We could implement the distribution as follows:

.. code-block:: python

    @metadist(implements="core.new_distribution", var_type="continuous", is_unique=True, version="1.0")
    class NewDistribution(BaseDistribution, UniqueDistributionMixin):
        """New custom distribution."""
        @classmethod
        def default_distribution(cls) -> BaseDistribution:
            return cls(0.0)

        @classmethod
        def _param_schema(cls):
            return {
                "value": {"type": "number"}
            }

        def _fit(self, data):
            # Implement your fitting logic here
            pass

        def draw(self):
            # Implement your drawing logic here
            pass

        def _param_dict(self):
            # Implement your parameter dictionary logic here
            pass

And then add it to the BuiltinDistributionProvider list in the :mod:`~metasyn.distribution.provider` module:

.. code-block:: python

    import NewDistribution

    class BuiltinDistributionProvider(BaseDistributionProvider):
    """Distribution tree that includes the builtin distributions."""

    name = "builtin"
    version = "1.1"
    distributions = [
        # ... other distributions
        NewDistribution,
    ]



