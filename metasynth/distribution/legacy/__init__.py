"""Legacy distributions that will be removed in the future."""

from metasynth.distribution.legacy.regex import RegexDistribution as Regex1_0
from metasynth.distribution.legacy.regex import UniqueRegexDistribution as UniqueRegex1_0


__all__ = ["Regex1_0", "UniqueRegex1_0"]
