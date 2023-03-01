"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasynth.distribution.base import DiscreteDistribution
from metasynth.distribution.base import ContinuousDistribution
from metasynth.distribution.base import CategoricalDistribution
from metasynth.distribution.base import DateDistribution
from metasynth.distribution.base import DateTimeDistribution
from metasynth.distribution.base import StringDistribution
from metasynth.distribution.base import TimeDistribution
from metasynth.distribution.categorical import MultinoulliDistribution
from metasynth.distribution.continuous import UniformDistribution
from metasynth.distribution.continuous import NormalDistribution
from metasynth.distribution.continuous import LogNormalDistribution
from metasynth.distribution.continuous import TruncatedNormalDistribution
from metasynth.distribution.continuous import ExponentialDistribution
from metasynth.distribution.datetime import UniformDateDistribution
from metasynth.distribution.datetime import UniformDateTimeDistribution
from metasynth.distribution.datetime import UniformTimeDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution
from metasynth.distribution.discrete import PoissonDistribution
from metasynth.distribution.discrete import UniqueKeyDistribution
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.regex import RegexDistribution
from metasynth.distribution.regex import UniqueRegexDistribution


__all__ = [
    "DiscreteDistribution", "ContinuousDistribution", "CategoricalDistribution",
    "DateDistribution", "DateTimeDistribution", "StringDistribution", "TimeDistribution",
    "MultinoulliDistribution", "UniformDistribution", "NormalDistribution",
    "LogNormalDistribution", "TruncatedNormalDistribution", "ExponentialDistribution",
    "DiscreteUniformDistribution", "PoissonDistribution", "UniqueKeyDistribution",
    "UniformDateDistribution", "UniformDateTimeDistribution", "UniformTimeDistribution",
    "FakerDistribution", "RegexDistribution", "UniqueRegexDistribution",
]
