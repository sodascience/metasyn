"""Module with discrete distributions."""

import numpy as np
from scipy.stats import randint

from metasynth.distribution.base import ScipyDistribution


class DiscreteUniformDistribution(ScipyDistribution):
    """Integer uniform distribution.

    It differs from the floating point uniform distribution by
    being a discrete distribution instead.

    Parameters
    ----------
    low: int
        Lower bound (inclusive) of the uniform distribution.
    high: int
        Upper bound (exclusive) of the uniform distribution.
    """

    dist_class = randint

    def __init__(self, low, high):
        self.par = {"low": low, "high": high}
        self.dist = self.dist_class(low=low, high=high)

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpmf(vals.values.astype(int)+1e-7))

    @classmethod
    def _fit(cls, values):
        param = {"low": np.min(values), "high": np.max(values)+1}
        return cls(**param)
