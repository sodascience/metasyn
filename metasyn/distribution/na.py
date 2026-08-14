"""Module implementing NA distributions.

This module contains a single class for creating distributions that only
return NA.
"""


import numpy as np

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    builtin_fitter,
    convert_to_series,
    metadist,
)


@metadist(name="core.na", var_type=["continuous", "discrete", "categorical", "string"])
class NADistribution(BaseDistribution):
    """Distribution that always returns NA values (None)."""

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls()

    def draw(self):
        return None

    def _param_dict(self):
        return {}

    @classmethod
    def _param_schema(cls):
        return {}

    def information_criterion(self, values): # noqa: ARG002
        series = convert_to_series(values)
        if len(series) == 0:
            return -np.inf
        return np.inf

@builtin_fitter(distribution=NADistribution,
                var_type=["continuous", "discrete", "categorical", "string"])
class NAFitter(BaseFitter):
    """Fitter for NA distribution."""

    def _fit(self, series, fit_log):  # noqa: ARG002
        return self.distribution()
