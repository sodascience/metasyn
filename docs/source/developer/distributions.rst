Distributions 
=============

This page is intended to provide an overview of how distributions are implemented and organized in ``metasyn``. It should help you understand how to create new distributions, or modify existing ones. 

For a detailed overview of classes, methods and attributes mentioned on this page, refer to the :doc:`/api/metasyn`. Clicking on object names will automatically take you to their API reference page.

Distribution subpackage
------------------------

Classes used to represent and handle distributions are located in the :mod:`~metasyn.distribution` subpackage. This subpackage contains modules used to represent different types of distributions, as well as a :mod:`~metasyn.distribution.base`  module that contains the base classes for all distributions.

Base submodule
~~~~~~~~~~~~~~
The :mod:`~metasyn.distribution.base` module contains the :class:`~metasyn.distribution.base.BaseDistribution` class, which is the base class for all distributions. It also contains the :class:`~metasyn.distribution.base.ScipyDistribution` class, which is a specialized base class for distributions that are built on top of SciPy's statistical distributions. 

Additionally it contains the :class:`~metasyn.distribution.base.UniqueDistributionMixin` class, which is a mixin class that can be used to make a distribution unique (i.e., one that does not contain duplicate values).

Finally it contains the :func:`~metasyn.distribution.base.metadist` decorator, which is used to set the attributes of a distribution.

BaseDistribution class
^^^^^^^^^^^^^^^^^^^^^^
This is the base class providing the basic structure for all distributions. It is not intended to be used directly, but rather to be derived from when implementing a new distribution.

:class:`~metasyn.distribution.base.BaseDistribution` has the following attributes:

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
| unique     | A boolean indicating whether the values in the distribution are unique. | ``True``, ``False``                         |
+------------+-------------------------------------------------------------------------+---------------------------------------------+
| version    | The version of the distribution.                                        | ``"1.0"``, ``"2.3"``, etc.                  |
+------------+-------------------------------------------------------------------------+---------------------------------------------+

Though they can be set manually, the intended way of setting these attributes is through the use of the :func:`~metasyn.distribution.base.metadist` decorator (which is covered further below).

.. admonition:: Note on 'implements' attributes

    The naming convention for the ``implements`` attribute is: 
    
    ``<prefix>.<distribution_name>``
    
    Distributions that are part of the core metasyn distribution provider list should use ``core`` as the prefix, e.g. ``core.multinoulli``.

BaseDistribution class has a series of abstract methods that *must be* implemeted by derived classes, these are:

- :meth:`~metasyn.distribution.base.BaseDistribution._fit` to contain the fitting logic for the distribution. It does not need to handle N/A values. 
- :meth:`~metasyn.distribution.base.BaseDistribution.draw` to draw a new value from the distribution.
- :meth:`~metasyn.distribution.base.BaseDistribution._param_dict` to return a dictionary of the distribution's parameters. 
- :meth:`~metasyn.distribution.base.BaseDistribution._param_schema` to return a schema for the distribution's parameters. 
- :meth:`~metasyn.distribution.base.BaseDistribution.default_distribution` to return a distribution with default parameters.

If the distribution has subsequently draws that are not independent, it is recommended to implement :meth:`~metasyn.distribution.base.BaseDistribution.draw_reset`. As the name suggests, this method is intended to reset the distribution's drawing mechanism.

It is recommended to also implement :meth:`~metasyn.distribution.base.BaseDistribution.information_criterion`. This is a class method used to determine which distribution gets selected during the fitting process for a series of values. The distribution with the lowest information criterion of the correct variable type will be selected. For discrete and continuous distributions it is currently implemented as `BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_). 

.. warning:: 

    Despite not being an abstract method in :class:`~metasyn.distribution.base.BaseDistribution`, it is recommended to implement a constructor (``__init__``) method in derived classes to initialize a distribution with a set of (distribution specific) parameters. 

There are more methods, but this is a good starting point when implementing a new distribution.
For an overview of the rest of the methods and implementation details, refer to the :class:`~metasyn.distribution.base.BaseDistribution` class.



Metadist decorator method
^^^^^^^^^^^^^^^^^^^^^^^^^
When implementing a new distribution (that inherits from :class:`~metasyn.distribution.base.BaseDistribution`), the :func:`~metasyn.distribution.base.metadist` decorator is intended to be used to set its attributes. 

To use the decorator, annotate a distribution class with ``@metadist``, passing in the attributes of the target distribution as parameters.

For example, the following distributions use the decorator as follows:

.. code-block:: python

    @metadist(implements="core.multinoulli", var_type=["categorical", "discrete", "string"])
    class MultinoulliDistribution(BaseDistribution):

.. code-block:: python

    @metadist(implements="core.regex", var_type="string", unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python
      
    @metadist(implements="core.uniform", var_type="date")
    class UniformDateDistribution(BaseUniformDistribution):


The :func:`~metasyn.distribution.base.metadist` decorator, which is a part of the :mod:`metasyn.distribution.base` submodule, is directly accessible when importing the main metasyn package, as it's explicitly and relatively imported upon importing the main metasyn package.

ScipyDistribution class
^^^^^^^^^^^^^^^^^^^^^^^
The :class:`~metasyn.distribution.base.ScipyDistribution` is a specialized base class for distributions that are based on
`SciPy <https://docs.scipy.org/doc/scipy/index.html>`_ statistical distributions. 

All the current :mod:`~metasyn.distribution.discrete` and :mod:`~metasyn.distribution.continuous` distributions are derived from this class.


UniqueDistributionMixin class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`~metasyn.distribution.base.UniqueDistributionMixin` is a mixin class that can be combined with other distribution classes to create distributions that generate unique values.

For example, the unique variants of the :class:`~metasyn.distribution.regex.RegexDistribution` and the :class:`~metasyn.distribution.faker.UniqueFakerDistribution` are implemented using this mixin as follows:

.. code-block:: python

    @metadist(implements="core.regex", var_type="string", unique=True)
    class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):

.. code-block:: python

    @metadist(implements="core.faker", var_type="string")
    class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):

.. warning:: 
    
    This mixin class has a default implementation that will work for many distributions, but it may not be appropriate for all. Be sure to check the implementation before using it. 


Other modules
~~~~~~~~~~~~~
The rest of the modules in the :mod:`~metasyn.distribution` subpackage contain the classes used to represent different types of distributions. A comprehensive overview of these modules, along with the distributions they implement, can be found on the API reference's :doc:`/api/metasyn.distribution` page.


Creating a new distribution
---------------------------
The first step to creating a new distribution is to inherit from a distribution class. This can be a base class (e.g. :class:`~metasyn.distribution.base.BaseDistribution`, :class:`~metasyn.distribution.base.ScipyDistribution`), or an existing distribution.

The next step is to set the attributes of the distribution using the :func:`~metasyn.distribution.base.metadist` decorator. Refer to :class:`~metasyn.distribution.base.BaseDistribution` for an overview of these attributes.

.. admonition:: Important

    In is posible to have different variations of the same distribution, for various data types. As is the case with the ``core.uniform`` distributions in ``metasyn``. 

Then, implement the required methods (:meth:`~metasyn.distribution.base.BaseDistribution._fit`, :meth:`~metasyn.distribution.base.BaseDistribution.draw`, :meth:`~metasyn.distribution.base.BaseDistribution._param_dict`, :meth:`~metasyn.distribution.base.BaseDistribution._param_schema`, :meth:`~metasyn.distribution.base.BaseDistribution.default_distribution` and ``__init__``), as well as any other applicable methods. 

Finally the distribution has to be added to a provider list, so that it can be used by ``metasyn`` for fitting.

For example, let's say we want to create a new distribution for unique continuous variables, to be a part of the core ``metasyn`` distribution provider. We could implement the distribution as follows:

.. code-block:: python

    @metadist(implements="core.new_distribution", var_type="continuous", unique=True, version="1.0")
    class NewDistribution(UniqueDistributionMixin, BaseDistribution):
        """New custom distribution."""

        def __init__(self, lower=0, upper=1):
            self.lower = lower
            self.upper = upper

        @classmethod
        def default_distribution(cls): 
            return cls(0, 1) # default distribution with lower=0 and upper=1

        @classmethod
        def _param_schema(cls):
            return {
                "lower": {"type": "number"},
                "upper": {"type": "number"},
            }

        @classmethod
        def _fit(cls, values):
            lower = min(values)
            upper = max(values)
            return cls(lower, upper)

        def draw(self):
            return random.uniform(self.lower, self.upper)

        def _param_dict(self):
            return {"lower": self.lower, "upper": self.upper}



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


Note that this is a bare-bones example and that the implementation of the distribution will vary depending on the type of distribution being implemented. 

Creating a distribution plugin
------------------------------
In case you want to create a new distribution as part of an add-on, as opposed to it being implemented in the core package, you can easily do so by following the available `distribution plugin template <https://github.com/sodascience/metasyn-distribution-template>`_.

More information on creating plug-ins can be found in the :doc:`/developer/plugins` section of the documentation.
