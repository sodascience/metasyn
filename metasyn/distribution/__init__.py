"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.continuous import (
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)
from metasyn.distribution.datetime import (
    UniformDateDistribution,
    UniformDateTimeDistribution,
    UniformTimeDistribution,
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
from metasyn.distribution.constant import (
    ConstantDistribution,
    DiscreteConstantDistribution,
    StringConstantDistribution,
)

__all__ = [
    "MultinoulliDistribution",
    "ExponentialDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
    "UniformDateDistribution",
    "UniformDateTimeDistribution",
    "UniformTimeDistribution",
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
]
