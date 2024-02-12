"""Package providing different distributions used in metasyn.

The package consists of several distribution modules, it also includes
``base`` module which forms the basis of all distributions.

Each distribution class provides methods for fitting the distribution to a
a series of values, and for generating synthetic data based on the fitted
distribution. Each distribution class also provides a way to calculate the
information criterion, used for selecting the best distribution for
a given set of values.
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
