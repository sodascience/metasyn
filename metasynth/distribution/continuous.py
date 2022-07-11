"""Implemented floating point distributions."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform, norm, lognorm, truncnorm
from scipy.stats._continuous_distns import FitDataError

from metasynth.distribution.base import ScipyDistribution, ContinuousDistribution


class UniformDistribution(ScipyDistribution, ContinuousDistribution):
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

    aliases = ["UniformDistribution", "uniform"]
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
        if np.any(np.array(values) < self.min_val) or np.any(np.array(values) > self.max_val):
            return 2*self.n_par + 100*len(vals)
        return 2*self.n_par - 2*len(vals)*np.log((self.max_val-self.min_val)**-1)


class NormalDistribution(ScipyDistribution, ContinuousDistribution):
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

    aliases = ["NormalDistribution", "normal", "gaussian"]
    dist_class = norm

    def __init__(self, mean, std_dev):
        self.par = {"mean": mean, "std_dev": std_dev}
        self.dist = norm(loc=mean, scale=std_dev)


class LogNormalDistribution(ScipyDistribution, ContinuousDistribution):
    """Log-normal distribution for floating point type.

    This class implements the log-normal mu and sigma as initialization input.

    Parameters
    ----------
    sigma: float
        Controls the width of the distribution.
    mu: float
        Controls the mean of the distribution.
    """

    aliases = ["LogNormalDistribution", "lognormal"]
    dist_class = lognorm

    def __init__(self, mu: float, sigma: float):  # pylint: disable=invalid-name
        self.par = {"mu": mu, "sigma": sigma}
        self.dist = lognorm(s=sigma, scale=np.exp(mu))

    @classmethod
    def _fit(cls, values):
        try:
            sigma, _, scale = cls.dist_class.fit(values, floc=0)
        except FitDataError:
            return cls(0, 1)
        return cls(np.log(scale), sigma)


class TruncatedNormalDistribution(ScipyDistribution, ContinuousDistribution):
    """Truncated normal distribution for floating point type.

    Parameters
    ----------
    lower_bound: float
        Lower bound of the truncated normal distribution.
    upper_bound: float
        Upper bound of the truncated normal distribution.
    mu: float
        Mean of the non-truncated normal distribution.
    sigma: float
        Standard deviation of the non-truncated normal distribution.
    """

    aliases = ["TruncatedNormalDistribution", "truncnormal", "boundednormal",
               "truncatednormal"]
    dist_class = truncnorm

    def __init__(self, lower_bound: float, upper_bound: float,
                 mu: float, sigma: float):
        self.par = {"lower_bound": lower_bound, "upper_bound": upper_bound,
                    "mu": mu, "sigma": sigma}
        a, b = (lower_bound-mu)/sigma, (upper_bound-mu)/sigma
        self.dist = truncnorm(a=a, b=b, loc=mu, scale=sigma)

    @classmethod
    def _fit(cls, values):
        lower_bound = np.min(values) - 1e-8
        upper_bound = np.max(values) + 1e-8
        return cls._fit_with_bounds(values, lower_bound, upper_bound)

    @classmethod
    def _fit_with_bounds(cls, values, lower_bound, upper_bound):
        def minimizer(param):
            mu, sigma = param
            a, b = (lower_bound-mu)/sigma, (upper_bound-mu)/sigma
            dist = truncnorm(a=a, b=b, loc=mu, scale=sigma)
            return -np.sum(dist.logpdf(values))

        x_start = [(lower_bound+upper_bound)/2, (upper_bound-lower_bound)/4]
        mu, sigma = minimize(minimizer, x_start,
                             bounds=[(None, None),
                                     ((upper_bound-lower_bound)/100, None)]).x
        return cls(lower_bound, upper_bound, mu, sigma)
