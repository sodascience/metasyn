Distribution list
=================

This page serves as an overview of the various distributions available in the
``metasyn.distribution`` module. Clicking on a distribution will take you to a page with more information, including its parameters.

Categorical Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: metasyn.distribution.categorical

.. autosummary::
   :toctree: generated/

   MultinoulliDistribution -- Stores labels and probabilities.


Continuous Distributions
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: metasyn.distribution.continuous

.. autosummary::
   :toctree: generated/

   UniformDistribution
   NormalDistribution
   LogNormalDistribution
   TruncatedNormalDistribution
   ExponentialDistribution
   ConstantDistribution

Date and Time Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: metasyn.distribution.datetime

.. autosummary::
   :toctree: generated/

   DateUniformDistribution
   TimeUniformDistribution
   DateTimeUniformDistribution
   DateTimeConstantDistribution
   TimeConstantDistribution
   DateConstantDistribution


Discrete Distributions
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: metasyn.distribution.discrete

.. autosummary::
   :toctree: generated/

   DiscreteUniformDistribution
   DiscreteNormalDistribution
   DiscreteTruncatedNormalDistribution
   PoissonDistribution
   UniqueKeyDistribution
   DiscreteConstantDistribution -- Constant discrete distribution

String Distributions
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: metasyn.distribution.string

.. autosummary::
   :toctree: generated/
   
   FakerDistribution
   UniqueFakerDistribution
   FreeTextDistribution
   StringConstantDistribution
   RegexDistribution
   UniqueRegexDistribution
