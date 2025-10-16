"""Module implementing the Poisson distribution."""

import numpy as np
from scipy.stats import poisson

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    ScipyDistribution,
    builtin_fitter,
    metadist,
)


@metadist(name="core.poisson", var_type="discrete")
class PoissonDistribution(ScipyDistribution):
    """Poisson distribution.

    Parameters
    ----------
    rate:
        Rate (mean) of the poisson distribution.

    Examples
    --------
    >>> PoissonDistribution(rate=3.5)
    """

    dist_class = poisson

    def __init__(self, rate: float):
        self.par = {"rate": rate}
        self.dist = self.dist_class(mu=rate)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls(0.5)

    @classmethod
    def _param_schema(cls):
        return {
            "rate": {"type": "number"},
        }


@builtin_fitter(distribution=PoissonDistribution, var_type="discrete")
class PoissonFitter(BaseFitter):
    """Fitter for Poisson distribution."""

    def _fit(self, series):
        mean_series = series.mean()
        mean_series = mean_series if mean_series >= 0 else 0
        return self.distribution(mean_series)
