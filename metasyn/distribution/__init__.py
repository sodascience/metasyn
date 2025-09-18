"""Package providing different distributions used in metasyn.

The package consists of several distribution modules, it also includes
``base`` module which forms the basis of all distributions.

Each distribution class provides methods for fitting the distribution to a
a series of values, and for generating synthetic data based on the fitted
distribution. Each distribution class also provides a way to calculate the
information criterion, used for selecting the best distribution for
a given set of values.
"""

from metasyn.distribution.categorical import MultinoulliDistribution, MultinoulliFitter
from metasyn.distribution.constant import (
    ContinuousConstantDistribution,
    ContinuousConstantFitter,
    DateConstantDistribution,
    DateConstantFitter,
    DateTimeConstantDistribution,
    DateTimeConstantFitter,
    DiscreteConstantDistribution,
    DiscreteConstantFitter,
    StringConstantDistribution,
    StringConstantFitter,
    TimeConstantDistribution,
    TimeConstantFitter,
)
from metasyn.distribution.exponential import ExponentialDistribution, ExponentialFitter
from metasyn.distribution.faker import (
    FakerDistribution,
    FakerFitter,
    UniqueFakerDistribution,
    UniqueFakerFitter,
)
from metasyn.distribution.freetext import FreeTextDistribution, FreeTextFitter
from metasyn.distribution.na import NADistribution, NAFitter
from metasyn.distribution.normal import (
    ContinuousNormalDistribution,
    ContinuousNormalFitter,
    ContinuousTruncatedNormalDistribution,
    ContinuousTruncatedNormalFitter,
    DiscreteNormalDistribution,
    DiscreteNormalFitter,
    DiscreteTruncatedNormalDistribution,
    DiscreteTruncatedNormalFitter,
    LogNormalDistribution,
    LogNormalFitter,
)
from metasyn.distribution.poisson import PoissonDistribution, PoissonFitter
from metasyn.distribution.regex import (
    RegexDistribution,
    RegexFitter,
    UniqueRegexDistribution,
    UniqueRegexFitter,
)
from metasyn.distribution.uniform import (
    ContinuousUniformDistribution,
    ContinuousUniformFitter,
    DateTimeUniformDistribution,
    DateTimeUniformFitter,
    DateUniformDistribution,
    DateUniformFitter,
    DiscreteUniformDistribution,
    DiscreteUniformFitter,
    TimeUniformDistribution,
    TimeUniformFitter,
)
from metasyn.distribution.uniquekey import UniqueKeyFitter

builtin_fitters = [
    DiscreteUniformFitter, ContinuousUniformFitter, DateUniformFitter, TimeUniformFitter,
    DateTimeUniformFitter,
    RegexFitter, UniqueRegexFitter,
    ContinuousConstantFitter, DiscreteConstantFitter, DateConstantFitter,
    DateTimeConstantFitter, TimeConstantFitter, StringConstantFitter,
    ExponentialFitter,
    MultinoulliFitter,
    FakerFitter, UniqueFakerFitter,
    FreeTextFitter,
    PoissonFitter,
    ContinuousNormalFitter, LogNormalFitter, DiscreteTruncatedNormalFitter,
    ContinuousTruncatedNormalFitter, DiscreteNormalFitter,
    UniqueKeyFitter,
    NAFitter,
]

__all__ = [
    "MultinoulliDistribution",
    "ExponentialDistribution",
    "LogNormalDistribution",
    "ContinuousNormalDistribution",
    "ContinuousTruncatedNormalDistribution",
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
