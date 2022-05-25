"""Implemented floating point distributions."""

import numpy as np
from scipy.stats import uniform, norm

from metasynth.distribution.base import ScipyDistribution


class UniformDistribution(ScipyDistribution):
    """Uniform distribution for floating point type.

    This class implements the uniform distribution between a minimum
    and maximum.

    Parameters
    ----------
    min_val: float
        Lower bound for uniform distribution.
    max_val: float
        Upper bound for uniform distribution.
    """

    aliases = ["uniform"]
    dist_class = uniform

    def __init__(self, min_val, max_val):
        self.par = {"min_val": min_val, "max_val": max_val}
        self.dist = uniform(loc=self.min_val, scale=self.max_val-self.min_val)

    @classmethod
    def _fit(cls, values):
        vals = values[~np.isnan(values)]
        return cls(vals.min(), vals.max())

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        if np.any(np.array(values) < self.min_val) and np.all(np.array(values) > self.max_val):
            return 2*self.n_par + 100*len(vals)
        return 2*self.n_par - 2*len(vals)*np.log((self.max_val-self.min_val)**-1)


class NormalDistribution(ScipyDistribution):
    """Normal distribution for floating point type.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    std_dev: float
        Standard deviation of the normal distribution.
    """

    aliases = ["normal", "gaussian"]
    dist_class = norm

    def __init__(self, mean, std_dev):
        self.par = {"mean": mean, "std_dev": std_dev}
        self.dist = norm(loc=mean, scale=std_dev)
