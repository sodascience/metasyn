"""Module implementing continuous (floating point) distributions."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, lognorm, norm, truncnorm, uniform
from scipy.stats._continuous_distns import FitDataError

from metasyn.distribution.base import ScipyDistribution, metadist


@metadist(implements="core.uniform", var_type="continuous")
class UniformDistribution(ScipyDistribution):
    """Uniform distribution for floating point type.

    This class implements the uniform distribution between a minimum
    and maximum.

    Parameters
    ----------
    lower: float
        Lower bound for uniform distribution.
    upper: float
        Upper bound for uniform distribution.
    """

    dist_class = uniform

    def __init__(self, lower: float, upper: float):
        self.par = {"lower": lower, "upper": upper}
        self.dist = uniform(loc=self.lower, scale=max(self.upper-self.lower, 1e-8))

    @classmethod
    def _fit(cls, values):
        return cls(values.min(), values.max())

    def _information_criterion(self, values):
        if np.any(np.array(values) < self.lower) or np.any(np.array(values) > self.upper):
            return np.log(len(values))*self.n_par + 100*len(values)
        if np.fabs(self.upper-self.lower) < 1e-8:
            return np.log(len(values))*self.n_par - 100*len(values)
        return (np.log(len(values))*self.n_par
                - 2*len(values)*np.log((self.upper-self.lower)**-1))

    @classmethod
    def default_distribution(cls):
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "number"},
            "upper": {"type": "number"},
        }


@metadist(implements="core.normal", var_type="continuous")
class NormalDistribution(ScipyDistribution):
    """Normal distribution for floating point type.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    sd: float
        Standard deviation of the normal distribution.
    """

    implements = "core.normal"
    dist_class = norm

    def __init__(self, mean: float, sd: float):
        self.par = {"mean": mean, "sd": sd}
        self.dist = norm(loc=mean, scale=max(sd, 1e-8))

    @classmethod
    def default_distribution(cls):
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }


@metadist(implements="core.lognormal", var_type="continuous")
class LogNormalDistribution(ScipyDistribution):
    """Log-normal distribution for floating point type.

    This class implements the log-normal mu and sigma as initialization input.

    Parameters
    ----------
    mean: float
        Controls the mean of the distribution.
    sd: float
        Controls the width of the distribution.
    """

    dist_class = lognorm

    def __init__(self, mean: float, sd: float):  # pylint: disable=invalid-name
        self.par = {"mean": mean, "sd": sd}
        self.dist = lognorm(s=max(sd, 1e-8), scale=np.exp(mean))

    @classmethod
    def _fit(cls, values):
        try:
            sd, _, scale = cls.dist_class.fit(values, floc=0)
        except FitDataError:
            return cls(0, 1)
        return cls(np.log(scale), sd)

    @classmethod
    def default_distribution(cls):
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }


@metadist(implements="core.truncated_normal", var_type="continuous")
class TruncatedNormalDistribution(ScipyDistribution):
    """Truncated normal distribution for floating point type.

    Parameters
    ----------
    lower: float
        Lower bound of the truncated normal distribution.
    upper: float
        Upper bound of the truncated normal distribution.
    mean: float
        Mean of the non-truncated normal distribution.
    sd: float
        Standard deviation of the non-truncated normal distribution.
    """

    dist_class = truncnorm

    def __init__(self, lower: float, upper: float,
                 mean: float, sd: float):
        self.par = {"lower": lower, "upper": upper,
                    "mean": mean, "sd": sd}
        a, b = (lower-mean)/sd, (upper-mean)/sd
        self.dist = truncnorm(a=a, b=b, loc=mean, scale=max(sd, 1e-8))

    @classmethod
    def _fit(cls, values):
        lower = values.min() - 1e-8
        upper = values.max() + 1e-8
        return cls._fit_with_bounds(values, lower, upper)

    @classmethod
    def _fit_with_bounds(cls, values, lower, upper):
        def minimizer(param):
            mean, sd = param
            a, b = (lower-mean)/sd, (upper-mean)/sd
            dist = truncnorm(a=a, b=b, loc=mean, scale=sd)
            return -np.sum(dist.logpdf(values))

        x_start = [(lower+upper)/2, (upper-lower)/4]
        mean, sd = minimize(minimizer, x_start,
                             bounds=[(None, None),
                                     ((upper-lower)/100, None)]).x
        return cls(lower, upper, mean, sd)

    @classmethod
    def default_distribution(cls):
        return cls(-1, 2, 0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "number"},
            "upper": {"type": "number"},
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }


@metadist(implements="core.exponential", var_type="continuous")
class ExponentialDistribution(ScipyDistribution):
    """Exponential distribution for floating point type.

    This class implements the exponential distribution with the rate as its
    single parameter.

    Parameters
    ----------
    rate: float
        Rate of the exponential distribution. This is equal to 1/mean of the distribution.
    """

    dist_class = expon

    def __init__(self, rate: float):
        self.par = {"rate": rate}
        self.dist = expon(loc=0, scale=1/max(rate, 1e-8))

    @classmethod
    def _fit(cls, values):
        values = values.filter(values > 0)
        if len(values) == 0:
            return cls.default_distribution()
        return cls(rate=1/expon.fit(values, floc=0)[1])

    @classmethod
    def default_distribution(cls):
        return cls(1.0)

    @classmethod
    def _param_schema(cls):
        return {
            "rate": {"type": "number"}
        }
