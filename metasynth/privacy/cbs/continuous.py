"""Disclosure control implementations for continuous distributions."""

from metasynth.distribution.continuous import UniformDistribution
from metasynth.distribution.continuous import NormalDistribution
from metasynth.distribution.continuous import LogNormalDistribution
from metasynth.distribution.continuous import TruncatedNormalDistribution
from metasynth.privacy.cbs.utils import get_cbs_bounds


class CbsUniform(UniformDistribution):
    """Uniform distribution implementation."""
    @classmethod
    def _fit(cls, values, n_avg=5):
        return cls(*get_cbs_bounds(values, n_avg))


class CbsNormal(NormalDistribution):
    """Normal distribution implementation."""


class CbsLogNormal(LogNormalDistribution):
    """Log-normal distribution implementation."""


class CbsTruncatedNormal(TruncatedNormalDistribution):
    """Truncated normal distribution implementation."""
    @classmethod
    def _fit(cls, values, n_avg=5):
        cls._fit_with_bounds(values, *get_cbs_bounds(values, n_avg=n_avg))
