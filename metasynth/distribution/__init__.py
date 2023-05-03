"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasynth.distribution.base import (CategoricalDistribution,
                                         ContinuousDistribution,
                                         DateDistribution,
                                         DateTimeDistribution,
                                         DiscreteDistribution,
                                         StringDistribution, TimeDistribution)
from metasynth.distribution.categorical import MultinoulliDistribution
from metasynth.distribution.continuous import (ExponentialDistribution,
                                               LogNormalDistribution,
                                               NormalDistribution,
                                               TruncatedNormalDistribution,
                                               UniformDistribution)
from metasynth.distribution.datetime import (UniformDateDistribution,
                                             UniformDateTimeDistribution,
                                             UniformTimeDistribution)
from metasynth.distribution.discrete import (DiscreteUniformDistribution,
                                             PoissonDistribution,
                                             UniqueKeyDistribution)
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.regex import (RegexDistribution,
                                          UniqueRegexDistribution)

__all__ = [
    "DiscreteDistribution", "ContinuousDistribution", "CategoricalDistribution",
    "DateDistribution", "DateTimeDistribution", "StringDistribution", "TimeDistribution",
    "MultinoulliDistribution", "UniformDistribution", "NormalDistribution",
    "LogNormalDistribution", "TruncatedNormalDistribution", "ExponentialDistribution",
    "DiscreteUniformDistribution", "PoissonDistribution", "UniqueKeyDistribution",
    "UniformDateDistribution", "UniformDateTimeDistribution", "UniformTimeDistribution",
    "FakerDistribution", "RegexDistribution", "UniqueRegexDistribution",
]
