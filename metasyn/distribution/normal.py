"""Module implementing normal distributions."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, lognorm, norm, truncnorm, uniform
from scipy.stats._continuous_distns import FitDataError

from metasyn.distribution.base import (
    BaseConstantDistribution,
    BaseDistribution,
    ScipyDistribution,
    metadist,
)


@metadist(name="core.normal", var_type="continuous")
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

    Examples
    --------
    >>> NormalDistribution(mean=1.3, sd=4.5)
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

@metafit(distribution=NormalDistribution, var_type="continuous")
class ContinuousNormalFitter(ScipyFitter):
    pass

@metadist(name="core.lognormal", var_type="continuous")
class LogNormalDistribution(ScipyDistribution):
    """Log-normal distribution for floating point type.

    This class implements the log-normal mu and sigma as initialization input.

    Parameters
    ----------
    mean: float
        Controls the mean of the distribution.
    sd: float
        Controls the width of the distribution.

    Examples
    --------
    >>> LogNormalDistribution(mean=-2.0, sd=4.5)
    """

    dist_class = lognorm

    def __init__(self, mean: float, sd: float):
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


@metadist(name="core.truncated_normal", var_type="continuous")
class TruncatedNormalDistribution(ScipyDistribution):
    """Truncated normal distribution for floating point type.

    Parameters
    ----------
    lower:
        Lower bound of the truncated normal distribution.
    upper:
        Upper bound of the truncated normal distribution.
    mean:
        Mean of the non-truncated normal distribution.
    sd:
        Standard deviation of the non-truncated normal distribution.

    Examples
    --------
    >>> TruncatedNormalDistribution(lower=1.0, upper=3.5, mean=2.3, sd=5)
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



@metadist(name="core.normal", var_type="discrete")
class DiscreteNormalDistribution(NormalDistribution):
    """Normal discrete distribution.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    sd: float
        Standard deviation of the normal distribution.

    Examples
    --------
    >>> DiscreteNormalDistribution(mean=2.4, sd=1.2)
    """

    def draw(self):
        return int(super().draw())


@metafit(distribution=DiscreteNormalDistribution, var_type="discrete")
class DiscreteNormalFitter(ScipyFitter):
    pass


@metadist(name="core.truncated_normal", var_type="discrete")
class DiscreteTruncatedNormalDistribution(TruncatedNormalDistribution):
    """Truncated normal discrete distribution.

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

    Examples
    --------
    >>> DiscreteTruncatedNormalDistribution(lower=1.2, upper=4.5, mean=2.3, sd=4.5)
    """

    def draw(self):
        return int(super().draw())

@metafit(distribution=DiscreteTruncatedNormalDistribution, var_type="discrete")
class DiscreteTruncatedNormalFitter(ScipyFitter):
    pass