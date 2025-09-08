"""Module implementing NA distributions.

This module contains a single class for creating distributions that only
return NA.
"""

import polars as pl

from metasyn.distribution.base import BaseDistribution, metadist, metafit, BaseFitter


@metadist(name="core.na", var_type=["continuous", "discrete", "categorical", "string"])
class NADistribution(BaseDistribution):
    """Distribution that always returns NA values (None)."""

    @classmethod
    def _fit(cls, values: pl.Series) -> BaseDistribution: # noqa: ARG003
        return cls()

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls()

    def draw(self):
        return None

    def _param_dict(self):
        return {}

    @classmethod
    def _param_schema(cls):
        return {}

    def information_criterion(self, values): # noqa: ARG002
        return 1e10

@metafit(distribution=NADistribution, var_type=["continuous", "discrete", "categorical", "string"])
class NAFitter(BaseFitter):
    """Fitter for NA distribution."""

    def _fit(self, series):
        return self.distribution()
