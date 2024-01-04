"""
Module implementing constant distributions.

The module contains a base class for constant distributions, and subclasses
that implement constant distributions for different variable types.
"""

import datetime as dt

import numpy as np
import polars as pl
from numpy import Inf

from metasyn.distribution.base import BaseDistribution, metadist
from metasyn.distribution.datetime import convert_numpy_datetime


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



@metadist(implements="core.constant", var_type="discrete")
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


@metadist(implements="core.constant", var_type="string")
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


@metadist(implements="core.constant", var_type="datetime")
class DateTimeConstantDistribution(ConstantDistribution):
    """Constant distribution for datetime series."""

    def __init__(self, value):
        if isinstance(value, str):
            value = dt.datetime.fromisoformat(value)
        elif isinstance(value, np.datetime64):
            value = convert_numpy_datetime(value)
        super().__init__(value)

    def _param_dict(self):
        return {"value": self.value.isoformat()}

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("2022-07-15T10:39:36")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }


@metadist(implements="core.constant", var_type="time")
class TimeConstantDistribution(ConstantDistribution):
    """Constant distribution for time series."""

    def __init__(self, value):
        if isinstance(value, str):
            value = dt.time.fromisoformat(value)
        super().__init__(value)

    def _param_dict(self):
        return {"value": self.value.isoformat()}

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("10:39:36")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }


@metadist(implements="core.constant", var_type="date")
class DateConstantDistribution(ConstantDistribution):
    """Constant distribution for date series."""

    def __init__(self, value):
        if isinstance(value, str):
            value = dt.date.fromisoformat(value)
        super().__init__(value)

    def _param_dict(self):
        return {"value": self.value.isoformat()}

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("1903-07-15")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }
