"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from metasynth.distribution.categorical import CatFreqDistribution
from metasynth.distribution.continuous import NormalDistribution
from metasynth.distribution.continuous import UniformDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution
from metasynth.distribution.string import RegexDistribution
