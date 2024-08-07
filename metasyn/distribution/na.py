"""Module implementing NA distributions.

This module contains a single class for creating distributions that only
return NA.
"""

import polars as pl

from metasyn.distribution.base import BaseDistribution, metadist


@metadist(implements="core.na", var_type="string")
class NADistribution(BaseDistribution):
    """Distribution that always returns NA values (None)."""

    @classmethod
    def _fit(cls, values: pl.Series) -> BaseDistribution:
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

    def information_criterion(self, values):  # pylint: disable=unused-argument
        return 1e10
