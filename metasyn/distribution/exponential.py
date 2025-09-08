"""Module implementing continuous (floating point) distributions."""

from scipy.stats import expon

from metasyn.distribution.base import (
    ScipyDistribution,
    ScipyFitter,
    convert_to_series,
    metadist,
    metafit,
)


@metadist(name="core.exponential", var_type="continuous")
class ExponentialDistribution(ScipyDistribution):
    """Exponential distribution for floating point type.

    This class implements the exponential distribution with the rate as its
    single parameter.

    Parameters
    ----------
    rate:
        Rate of the exponential distribution. This is equal to 1/mean of the distribution.

    Examples
    --------
    >>> ExponentialDistribution(rate=2.4)
    """

    scipy_class = expon

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


@metafit(distribution=ExponentialDistribution, var_type="continuous")
class ExponentialFitter(ScipyFitter):
    """Fitter for exponential distribution."""

    def fit(self, values):
        series = convert_to_series(values)
        series = series.filter(series > 0)
        if len(series) == 0:
            return self.distribution.default_distribution()
        return self.distribution(rate=1/expon.fit(series, floc=0)[1])
