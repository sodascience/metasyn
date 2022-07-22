"""Module with the CBS implementations for discrete variables."""

import numpy as np

from metasynth.distribution.discrete import DiscreteUniformDistribution
from metasynth.distribution.discrete import PoissonDistribution
from metasynth.distribution.discrete import UniqueKeyDistribution
from metasynth.privacy.cbs.utils import get_cbs_bounds


class CbsDiscreteUniform(DiscreteUniformDistribution):
    """Implementation for discrete uniform distribution."""

    @classmethod
    def _fit(cls, values, n_avg: int=5):
        low, high = get_cbs_bounds(values, n_avg=n_avg)
        return cls(round(low), round(high))


class CbsPoisson(PoissonDistribution):
    """Implementation for poisson distribution."""


class CbsUniqueKey(UniqueKeyDistribution):
    """Implementation for unique key distribution."""

    @classmethod
    def _fit(cls, values, n_avg: int=5):
        orig_dist = super()._fit(values)
        if orig_dist.consecutive == 1:
            return cls(np.random.randint(2*n_avg)-n_avg, orig_dist.consecutive)
        uniform_dist = CbsDiscreteUniform.fit(values, n_avg=n_avg)
        return cls(uniform_dist.low, orig_dist.consecutive)  # type: ignore
