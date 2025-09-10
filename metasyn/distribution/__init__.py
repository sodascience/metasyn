"""Package providing different distributions used in metasyn.

The package consists of several distribution modules, it also includes
``base`` module which forms the basis of all distributions.

Each distribution class provides methods for fitting the distribution to a
a series of values, and for generating synthetic data based on the fitted
distribution. Each distribution class also provides a way to calculate the
information criterion, used for selecting the best distribution for
a given set of values.
"""

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.constant import (
    ContinuousConstantDistribution,
    DateConstantDistribution,
    DateTimeConstantDistribution,
    DiscreteConstantDistribution,
    StringConstantDistribution,
    TimeConstantDistribution,
)
from metasyn.distribution.exponential import ExponentialDistribution
from metasyn.distribution.faker import FakerDistribution, UniqueFakerDistribution
from metasyn.distribution.freetext import FreeTextDistribution
from metasyn.distribution.na import NADistribution
from metasyn.distribution.normal import (
    DiscreteNormalDistribution,
    DiscreteTruncatedNormalDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
)
from metasyn.distribution.poisson import PoissonDistribution
from metasyn.distribution.regex import (
    RegexDistribution,
    UniqueRegexDistribution,
)
from metasyn.distribution.uniform import (
    ContinuousUniformDistribution,
    DateTimeUniformDistribution,
    DateUniformDistribution,
    DiscreteUniformDistribution,
    TimeUniformDistribution,
)

__all__ = [
    "MultinoulliDistribution",
    "ExponentialDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "TruncatedNormalDistribution",
    "ContinuousUniformDistribution",
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
    "ContinuousConstantDistribution",
    "DiscreteConstantDistribution",
    "StringConstantDistribution",
    "DateConstantDistribution",
    "TimeConstantDistribution",
    "DateTimeConstantDistribution",
]
