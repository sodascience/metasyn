"""Module implementing normal distributions."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm, truncnorm
from scipy.stats._continuous_distns import FitDataError

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    ScipyDistribution,
    ScipyFitter,
    metadist,
    metafit,
)


@metadist(name="core.normal", var_type="continuous")
class ContinuousNormalDistribution(ScipyDistribution):
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

    scipy_class = norm

    def __init__(self, mean: float, sd: float):
        self.par = {"mean": mean, "sd": sd}
        self.dist = norm(loc=mean, scale=max(sd, 1e-8))

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }

@metafit(distribution=ContinuousNormalDistribution, var_type="continuous")
class ContinuousNormalFitter(ScipyFitter):
    """Fitter for continuous normal distribution."""

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

    scipy_class = lognorm

    def __init__(self, mean: float, sd: float):
        self.par = {"mean": mean, "sd": sd}
        self.dist = lognorm(s=max(sd, 1e-8), scale=np.exp(mean))

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }

@metafit(distribution=LogNormalDistribution, var_type="continuous")
class LogNormalFitter(BaseFitter):
    """Fitter for log normal distribution."""

    def _fit(self, series):
        try:
            sd, _, scale = self.distribution.scipy_class.fit(series, floc=0)
        except FitDataError:
            return self.distribution(0, 1)
        return self.distribution(np.log(scale), sd)


@metadist(name="core.truncated_normal", var_type="continuous")
class ContinuousTruncatedNormalDistribution(ScipyDistribution):
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

    scipy_class = truncnorm

    def __init__(self, lower: float, upper: float,
                 mean: float, sd: float):
        self.par = {"lower": lower, "upper": upper,
                    "mean": mean, "sd": sd}
        a, b = (lower-mean)/sd, (upper-mean)/sd
        self.dist = truncnorm(a=a, b=b, loc=mean, scale=max(sd, 1e-8))

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls(-1, 2, 0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "number"},
            "upper": {"type": "number"},
            "mean": {"type": "number"},
            "sd": {"type": "number"},
        }

@metafit(distribution=ContinuousTruncatedNormalDistribution, var_type="continuous")
class ContinuousTruncatedNormalFitter(BaseFitter):
    """Fitter for continuous truncated normal fitter."""

    def _fit(self, series):
        lower = series.min() - 1e-8
        upper = series.max() + 1e-8
        return self._fit_with_bounds(series, lower, upper)

    def _fit_with_bounds(self, values, lower, upper):
        def minimizer(param):
            mean, sd = param
            a, b = (lower-mean)/sd, (upper-mean)/sd
            dist = truncnorm(a=a, b=b, loc=mean, scale=sd)
            return -np.sum(dist.logpdf(values))

        x_start = [(lower+upper)/2, (upper-lower)/4]
        mean, sd = minimize(minimizer, x_start,
                             bounds=[(None, None),
                                     ((upper-lower)/100, None)]).x
        return self.distribution(lower, upper, mean, sd)


@metadist(name="core.normal", var_type="discrete")
class DiscreteNormalDistribution(ContinuousNormalDistribution):
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
    """Fitter for discrete normal distribution."""


@metadist(name="core.truncated_normal", var_type="discrete")
class DiscreteTruncatedNormalDistribution(ContinuousTruncatedNormalDistribution):
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
class DiscreteTruncatedNormalFitter(ContinuousTruncatedNormalFitter):
    """Fitter for discrete truncated normal distribution."""
