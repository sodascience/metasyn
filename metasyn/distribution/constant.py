"""Constant distributions and fitters."""
import datetime as dt

import numpy as np
import polars as pl

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    metadist,
    metafit,
)
from metasyn.distribution.util import convert_numpy_datetime
from metasyn.distribution.base import convert_to_series


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

    def _param_dict(self):
        return {"value": self.value}

    def draw(self):
        return self.value

    def information_criterion(self, values):
        vals = convert_to_series(values)
        return -np.inf if vals.n_unique() < 2 else np.inf


@metadist(name="core.constant", var_type="continuous")
class ContinuousConstantDistribution(BaseConstantDistribution):
    """Constant distribution for floating point type.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: float
        Value that will be returned when drawn.

    Examples
    --------
    >>> ConstantDistribution(2.45)
    """

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls(0.0)

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "number"}
        }


@metadist(name="core.constant", var_type="discrete")
class DiscreteConstantDistribution(BaseConstantDistribution):
    """Constant discrete distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: int
        Value that will be returned when drawn.

    Examples
    --------
    >>> DiscreteConstantDistribution(213456)
    """

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls(0)

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "integer"}
        }


class BaseConstantFitter(BaseFitter):
    """Base distribution for many constant fitters."""

    def _fit(self, values: pl.Series) -> BaseDistribution:
        # if unique, just get that value
        if values.n_unique() == 1:
            return self.distribution(values.unique()[0])

        # otherwise get most common value
        val = values.value_counts(sort=True)[0,0]
        return self.distribution(val)



@metadist(name="core.constant", var_type="datetime")
class DateTimeConstantDistribution(BaseConstantDistribution):
    """Constant datetime distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: str or datetime.datetime
        Value that will be returned when drawn.

    Examples
    --------
    >>> DateTimeConstantDistribution(value="2022-07-15T10:39:36")
    """

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


@metadist(name="core.constant", var_type="time")
class TimeConstantDistribution(BaseConstantDistribution):
    """Constant time distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: str or datetime.time
        Value that will be returned when drawn.

    Examples
    --------
    >>> TimeConstantDistribution(value="10:39:36")
    """

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


@metadist(name="core.constant", var_type="date")
class DateConstantDistribution(BaseConstantDistribution):
    """Constant date distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: str or datetime.date
        Value that will be returned when drawn.

    Examples
    --------
    >>> DateConstantDistribution(value="1903-07-15")
    """

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


@metadist(name="core.constant", var_type="string")
class StringConstantDistribution(BaseConstantDistribution):
    """Constant string distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: str
        Value that will be returned when drawn.

    Examples
    --------
    >>> ConstantDistribution("some_string")

    """

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("text")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }


@metafit(distribution=DiscreteConstantDistribution, var_type="discrete")
class DiscreteConstantFitter(BaseConstantFitter):
    """Fitter for constant discrete distribution."""

@metafit(distribution=ContinuousConstantDistribution, var_type="continuous")
class ContinuousConstantFitter(BaseConstantFitter):
    """Fitter for constant continuous distribution."""

@metafit(distribution=DateConstantDistribution, var_type="date")
class DateConstantFitter(BaseConstantFitter):
    """Fitter for constant date distribution."""

@metafit(distribution=TimeConstantDistribution, var_type="time")
class TimeConstantFitter(BaseConstantFitter):
    """Fitter for constant time distribution."""

@metafit(distribution=DateTimeConstantDistribution, var_type="datetime")
class DateTimeConstantFitter(BaseConstantFitter):
    """Fitter for constant datetime distribution."""

@metafit(distribution=StringConstantDistribution, var_type="string")
class StringConstantFitter(BaseConstantFitter):
    """Fitter for constant string distribution."""
