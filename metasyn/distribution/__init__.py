"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.continuous import (ExponentialDistribution,
                                             LogNormalDistribution,
                                             NormalDistribution,
                                             TruncatedNormalDistribution,
                                             UniformDistribution)
from metasyn.distribution.datetime import (UniformDateDistribution,
                                           UniformDateTimeDistribution,
                                           UniformTimeDistribution)
from metasyn.distribution.discrete import (DiscreteUniformDistribution,
                                           PoissonDistribution,
                                           UniqueKeyDistribution)
from metasyn.distribution.faker import (FakerDistribution, UniqueFakerDistribution,
                                        FreeTextDistribution)
from metasyn.distribution.regex import (RegexDistribution,
                                        UniqueRegexDistribution)
from metasyn.distribution.na import NADistribution
__all__ = [
    "MultinoulliDistribution", "UniformDistribution", "NormalDistribution",
    "LogNormalDistribution", "TruncatedNormalDistribution", "ExponentialDistribution",
    "DiscreteUniformDistribution", "PoissonDistribution", "UniqueKeyDistribution",
    "UniformDateDistribution", "UniformDateTimeDistribution", "UniformTimeDistribution",
    "FakerDistribution", "UniqueFakerDistribution", "RegexDistribution", "UniqueRegexDistribution",
    "NADistribution", "FreeTextDistribution"

]
