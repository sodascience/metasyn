Distributions 
=============

This page is intended to provide an overview of how distributions are implemented and organized in ``metasyn``. It should help you understand how to create new distributions, or modify existing ones. 

For a detailed overview of classes, methods and attributes mentioned on this page, refer to the :doc:`/api/metasyn`. Clicking on object names will automatically take you to their API reference page.

Distribution subpackage
------------------------

:mod:`~metasyn.distribution` is an essential subpackage that contains (submodules with) classes used to represent and handle various distributions. 

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

Finally it contains the :func:`~metasyn.distribution.metadist` decorator, which is used to set the attributes of a distribution.

BaseDistribution class
^^^^^^^^^^^^^^^^^^^^^^
This is the base class providing the basic structure for all distributions. It is not intended to be used directly, but rather to be derived from when implementing a new distribution.

:class:`~metasyn.distribution.BaseDistribution` has the following attributes:

+------------+-------------------------------------------------------------------------+---------------------------------------------+
| Attribute  | Description                                                             | Example(s)                                  |
+============+=========================================================================+=============================================+
| implements | A unique string identifier for the distribution type it implements      | ``"core.uniform"``, ``"user.months"``, etc. |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| var_type   | The type of variable associated with the distribution                   | ``"discrete"``, ``"continuous"``, etc.      |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| provenance | Information about the source (core, plugin, etc.) of the distribution.  | ``"builtin"``, ``"myplugin"``, etc.         |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| privacy    | The privacy class or implementation associated with the distribution.   | ``"none"``, ``"customPrivacyClass"``, etc.  |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| is_unique  | A boolean indicating whether the values in the distribution are unique. | ``True``, ``False``                         |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| version    | The version of the distribution.                                        | ``"1.0"``, ``"2.3"``, etc.                  |
+------------+-------------------------------------------------------------------------+---------------------------------------------+

Though they can be set manually, the intended way of setting these attributes is through the use of the ``metadist`` decorator (which is covered further below).

.. admonition:: Note on 'implements' attributes

    The naming convention for the ``implements`` attribute is: 
    
    ``<prefix>.<distribution_name>``
    
    Distributions that are part of the core metasyn distribution provider list should use ``core`` as the prefix, e.g. ``core.multinoulli``.

BaseDistribution class has a series of abstract methods that *must be* implemeted by derived classes, these are:

- :meth:`~metasyn.distribution.BaseDistribution._fit` to contain the fitting logic for the distribution. It does not need to handle N/A values. 
- :meth:`~metasyn.distribution.BaseDistribution.draw` to draw a new value from the distribution.
- :meth:`~metasyn.distribution.BaseDistribution._param_dict` to return a dictionary of the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution._param_schema` to return a schema for the distribution's parameters. 
- :meth:`~metasyn.distribution.BaseDistribution.default_distribution` to return a distribution with default parameters. 

If the distribution has subsequently draws that are not independent, it is recommended to implement :meth:`~metasyn.distribution.BaseDistribution.draw_reset`. As the name suggests, this method is intended to reset the distribution's drawing mechanism.

Additionally it is recommended to implement :meth:`~metasyn.distribution.BaseDistribution.information_criterion`. This is a class method used to determine which distribution gets selected during the fitting process for a series of values. The distribution with the lowest information criterion of the correct variable type will be selected. For discrete and continuous distributions it is currently implemented as `BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_). 

There are more methods, but this is a good starting point when implementing a new distribution.
For an overview of the rest of the methods and implementation details, refer to the :class:`~metasyn.distribution.BaseDistribution` class.


Metadist decorator method
^^^^^^^^^^^^^^^^^^^^^^^^^
When implementing a new distribution (that inherits from :class:`~metasyn.distribution.BaseDistribution`), the ``metadist`` decorator is intended to be used to set its attributes. 

To use the ``metadist`` decorator, annotate a distribution class with ``@metadist``, passing in the attributes of the target distribution as parameters.

For example, the following distributions use the decorator as follows:

.. code-block:: python

    @metadist(implements="core.multinoulli", var_type=["categorical", "discrete", "string"])
    class MultinoulliDistribution(BaseDistribution):

.. code-block:: python

    @metadist(implements="core.regex", var_type="string", is_unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python
      
    @metadist(implements="core.uniform", var_type="date")
    class UniformDateDistribution(BaseUniformDistribution):


The ``metadist`` decorator, which is a part of the :mod:`metasyn.distribution.base` submodule, is directly accessible when importing the main metasyn package, as it's explicitly and relatively imported upon importing the main metasyn package.

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

    @metadist(implements="core.regex", var_type="string", is_unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python

    @metadist(implements="core.faker", var_type="string")
    class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):

.. warning:: 
    
    This mixin class has a default implementation that will work for many distributions, but it may not be appropriate for all. Be sure to check the implementation before using it. 


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
The first step to creating a new distribution is to inherit from a distribution class. This can be a base class (e.g. :class:`~metasyn.distribution.BaseDistribution`, :class:`~metasyn.distribution.ScipyDistribution`), or an existing distribution.```

I think actually it might not work in some ways, since there might be some type checking that goes wrong.

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



