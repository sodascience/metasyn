"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.constant import (
    ConstantDistribution,
    DateConstantDistribution,
    DateTimeConstantDistribution,
    DiscreteConstantDistribution,
    StringConstantDistribution,
    TimeConstantDistribution,
)
from metasyn.distribution.continuous import (
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)
from metasyn.distribution.datetime import (
    DateTimeUniformDistribution,
    DateUniformDistribution,
    TimeUniformDistribution,
)
from metasyn.distribution.discrete import (
    DiscreteNormalDistribution,
    DiscreteTruncatedNormalDistribution,
    DiscreteUniformDistribution,
    PoissonDistribution,
    UniqueKeyDistribution,
)
from metasyn.distribution.faker import (
    FakerDistribution,
    FreeTextDistribution,
    UniqueFakerDistribution,
)
from metasyn.distribution.na import NADistribution
from metasyn.distribution.regex import RegexDistribution, UniqueRegexDistribution

__all__ = [
    "MultinoulliDistribution",
    "ExponentialDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
    "DateUniformDistribution",
    "DateTimeUniformDistribution",
    "TimeUniformDistribution",
    "DiscreteNormalDistribution",
    "DiscreteTruncatedNormalDistribution",
    "DiscreteUniformDistribution",
    "PoissonDistribution",
    "UniqueKeyDistribution",
    "FakerDistribution",
    "FreeTextDistribution",
    "UniqueFakerDistribution",
    "NADistribution",
    "RegexDistribution",
    "UniqueRegexDistribution",
    "ConstantDistribution",
    "DiscreteConstantDistribution",
    "StringConstantDistribution",
    "DateConstantDistribution",
    "TimeConstantDistribution",
    "DateTimeConstantDistribution",
]
