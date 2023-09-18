"""Module containing the class with NA distributions."""

import polars as pl

from metasyn.distribution.base import BaseDistribution, metadist


@metadist(implements="core.na", var_type="string")
class NADistribution(BaseDistribution):
    """Distribution that will only ever give back NA."""

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
