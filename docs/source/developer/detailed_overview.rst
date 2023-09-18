Detailed Overview
=================

This page features a detailed overview of metasyn's modules, with the goal of helping developers get started developing for metasyn.

.. warning:: 
  This page is unfinished and might be outdated, if information is lacking or does not seem right, feel free to :doc:`get in touch </about/contact>`  and we'll try to help you.

Metasyn File
--------------

The core of the metasyn is the Metasyn Dataset Format. This is a file in with the JSON format, that describes a *dataset*.
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

MetaFrame
---------

The main datastructure that the user will interface with. It contains exactly the information of the Metasyn File. The main
initialization methods are the classmethods :meth:`metasyn.metaframe.MetaFrame.from_json` (to load from a file) and 
:meth:`metasyn.metaframe.MetaFrame.from_dataframe` to fit variables from a
polars DataFrame. It implements ``__getitem__`` so that access to variables is easier, either by variable name or index.

The last important method is :meth:`metasyn.metaframe.MetaFrame.synthesize`, which creates a synthesized dataset.

MetaVar
-------

This is the variable level building block for the MetaFrame. It contains the methods to convert a polars `Series` into a 
variable with an appropriate distribution.

The variable can be manually created with direct initialization, but usually it is easier to use the
:meth:`metasyn.var.MetaVar.detect` method. This method does not fit any distribution, but it does infer the correct types for
the MetaVar and saves the `Series` for later fitting.

Next is the :meth:`metasyn.var.MetaVar.fit` method that actually fits a distribution to the data. Here you can again set the
distribution, privacy package and uniqueness for the variable.

Lastly, there is the :meth:`metasyn.var.MetaVar.draw_series` method that synthesizes a new series. For this to work,
the variable has to be fitted of course.

As can be inferred from the previous methods, the `MetaVar` class is to the `MetaFrame` what a polars `Series` is to a
`DataFrame`.

MetaDistribution
----------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each variable type a class is derived from the ``BaseDistribution``. It sets the ``var_type`` which is used in the ``MetaVar``
class and the Metasyn File. A distribution should always derive from one of those distributions, either directly or indirectly.

ScipyDistribution
~~~~~~~~~~~~~~~~~

This distribution is useful for discrete and continuous distributions that are based on
`SciPy <https://docs.scipy.org/doc/scipy/index.html>`_. Most of the currently implemented numerical distributions
use the ``ScipyDistribution`` as their base class (while also having either ``DiscreteDistribution`` or ``ContinuousDistribution``
as a baseclass).

:mod:`Privacy Features (experimental) <metasyn.privacy>`
----------------------------------------------------------

A system to incorporate privacy features such as differential privacy or other forms of disclosure control is being implemented.
This part of the code is considered to be particularly unstable, so modifications for future versions are likely necessary.

The :mod:`metasyn.privacy.cbs` sub-package is an example of how to implement a privacy package. Notice that all distributions
are derived from their non-private counterparts in :mod:`metasyn.distribution`. Only distributions that are derived in the
privacy package are available while fitting. Thus, if the privacy package simply wants the copy the distribution from the main
package it should simply use class derivation and add a docstring, such as :class:`metasyn.privacy.cbs.continuous.CbsNormal`.

The :mod:`metasyn.privacy.cbs` sub-package will be removed at some point and possibly be redistributed as its own package if
there is demand for it.



