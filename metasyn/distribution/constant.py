"""Module containing the class with constant distributions."""

import polars as pl
from numpy import Inf

from metasyn.distribution.base import BaseDistribution, metadist


class BaseConstantDistribution(BaseDistribution):
    """Base class for constant distribution.

    This base class makes it easy to implement new constant distributions
    for different variable types.
    """
    def __init__(self, value) -> None:
        self.value = value

    @property
    def n_par(self) -> int:
        """int: Number of parameters for distribution."""
        return 0

    @classmethod
    def _fit(cls, values: pl.Series) -> BaseDistribution:
        # if unique, just get that value
        if values.n_unique() == 1:
            return cls(values.unique()[0])

        # otherwise get most common value
        val = values.value_counts(sort=True)[0,0]
        return cls(val)

    def _param_dict(self):
        return {"value": self.value}

    def draw(self):
        return self.value

    def information_criterion(self, values):
        vals = self._to_series(values)
        return -Inf if vals.n_unique() < 2 else Inf


@metadist(implements="core.constant", var_type="continuous")
class ConstantDistribution(BaseConstantDistribution):
    """Constant distribution for continuous vars."""
    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls(0.0)

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "number"}
        }



@metadist(implements="core.discrete_constant", var_type="discrete")
class DiscreteConstantDistribution(BaseConstantDistribution):
    """Constant distribution for discrete vars."""
    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls(0)

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "integer"}
        }


@metadist(implements="core.string_constant", var_type="string")
class StringConstantDistribution(ConstantDistribution):
    """Constant distribution for string vars."""
    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("text")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }
