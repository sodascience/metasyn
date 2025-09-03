"""Uniform distributions and fitters."""

import datetime as dt
from abc import abstractmethod
from random import random
from typing import Any, Dict

import numpy as np
from scipy.stats import randint, uniform

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    ScipyDistribution,
    metadist,
    metafit,
)
from metasyn.distribution.util import convert_numpy_datetime


@metadist(name="core.uniform", var_type="discrete")
class DiscreteUniformDistribution(ScipyDistribution):
    """Uniform discrete distribution.

    It differs from the floating point uniform distribution by
    being a discrete distribution instead.

    Parameters
    ----------
    lower: int
        Lower bound (inclusive) of the uniform distribution.
    upper: int
        Upper bound (exclusive) of the uniform distribution.

    Examples
    --------
    >>> DiscreteUniformDistribution(lower=3, upper=20)
    """

    # dist = randint

    def __init__(self, lower: int, upper: int):
        self.par = {"lower": lower, "upper": upper}
        self.dist = randint(low=lower, high=upper)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))


    @classmethod
    def default_distribution(cls):
        return cls(0, 10)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "integer"},
            "upper": {"type": "integer"},
        }


@metafit(distribution=DiscreteUniformDistribution, var_type="discrete")
class DiscreteUniformFitter(BaseFitter):
    """Fitter for discrete uniform distribution."""

    def fit(self, values):
        return DiscreteUniformDistribution(values.min(), values.max()+1)




@metadist(name="core.uniform", var_type="continuous")
class ContinuousUniformDistribution(ScipyDistribution):
    """Uniform distribution for floating point type.

    This class implements the uniform distribution between a minimum
    and maximum.

    Parameters
    ----------
    lower: float
        Lower bound for uniform distribution.
    upper: float
        Upper bound for uniform distribution.

    Examples
    --------
    >>> UniformDistribution(lower=-3.0, upper=10.0)
    """

    # dist = uniform

    def __init__(self, lower: float, upper: float):
        self.par = {"lower": lower, "upper": upper}
        self.dist = uniform(loc=self.lower, scale=max(self.upper-self.lower, 1e-8))

    def _information_criterion(self, values):
        if np.any(np.array(values) < self.lower) or np.any(np.array(values) > self.upper):
            return np.log(len(values))*self.n_par + 100*len(values)
        if np.fabs(self.upper-self.lower) < 1e-8:
            return np.log(len(values))*self.n_par - 100*len(values)
        return (np.log(len(values))*self.n_par
                - 2*len(values)*np.log((self.upper-self.lower)**-1))

    @classmethod
    def default_distribution(cls):
        return cls(0, 1)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "number"},
            "upper": {"type": "number"},
        }

@metafit(distribution=ContinuousUniformDistribution, var_type="continuous")
class ContinuousUniformFitter(BaseFitter):
    """Fitter for continuous uniform distribution."""

    def fit(self, values):
        return ContinuousUniformDistribution(values.min(), values.max())


class BaseDTUniformDistribution(BaseDistribution):
    """Base class for all time/date/datetime uniform distributions."""

    precision_possibilities = ["microseconds", "seconds", "minutes", "hours", "days"]

    def __init__(self, lower: Any, upper: Any, precision: str = "microseconds"):
        if isinstance(lower, str):
            lower = self.fromisoformat(lower)
        elif isinstance(lower, np.datetime64):
            lower = convert_numpy_datetime(lower)
        if isinstance(upper, str):
            upper = self.fromisoformat(upper)
        elif isinstance(upper, np.datetime64):
            upper = convert_numpy_datetime(upper)
        self.precision = precision
        self.lower = self.round(lower)
        self.upper = self.round(upper)


    def round(self, time_obj: Any) -> Any:
        """Round down any time object with the precision.

        Parameters
        ----------
        time_obj:
            Object to round down.

        Return
        ------
        obj:
            Time/date/datetime object rounded down to the measured precision.
        """
        for prec in self.precision_possibilities:
            if prec == self.precision:
                break
            time_obj = time_obj.replace(**{prec[:-1]: 0})
        try:
            time_obj = time_obj.replace(nanosecond=0)
        except TypeError:
            pass
        return time_obj

    def draw(self) -> dt.datetime:
        delta = self.upper-self.lower + self.minimum_delta
        new_time = random()*delta + self.lower
        return self.round(new_time)

    @abstractmethod
    def fromisoformat(self, dt_obj: str):
        """Convert string to iso format."""

    @property
    def minimum_delta(self) -> dt.timedelta:
        """Get the minimum time delta."""
        return dt.timedelta(**{self.precision: 1})

    def information_criterion(self, values): # noqa: ARG002
        return 0.0

    def _param_dict(self):
        return {
            "lower": self.lower.isoformat(),
            "upper": self.upper.isoformat(),
            "precision": self.precision,
        }



@metadist(name="core.uniform", var_type="datetime")
class DateTimeUniformDistribution(BaseDTUniformDistribution):
    """Uniform DateTime distribution.

    Datetime objects will be uniformly distributed between a start and end date time.

    Parameters
    ----------
    lower: str or datetime.datetime
        Earliest possible datetime to be generated from this distribution.
        String formatted in ISO format, see examples below or use datetime objects
        from the python standard library `datetime`.
    upper: str or datetime.datetime
        Latest possible datetime to be generated from this distribution.
        String formatted in ISO format, see examples below or use datetime objects
        from the python standard library `datetime`.
    precision: str
        Most precise measure of datetime that the distribution will take into account.
        For example, if precision == "seconds", then the microseconds will always be
        set to 0. Possible values: ["microseconds", "seconds", "minutes", "hours", "days"].

    Examples
    --------
    >>> DateTimeUniformDistribution(lower="2022-07-15T10:39", upper="2022-08-15T10:39",
                                    precision="minutes")
    """

    def fromisoformat(self, dt_obj: str) -> dt.datetime:
        return dt.datetime.fromisoformat(dt_obj)

    @classmethod
    def default_distribution(cls):
        return cls("2022-07-15T10:39:36", "2022-08-15T10:39:36", precision="seconds")

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "string"},
            "upper": {"type": "string"},
            "precision": {"type": "string"},
        }


@metadist(name="core.uniform", var_type="time")
class TimeUniformDistribution(BaseDTUniformDistribution):
    """Uniform time distribution.

    Time objects will be uniformly distributed between a start and end time.

    Parameters
    ----------
    lower: str or datetime.time
        Earliest possible time to be generated from this distribution.
        String formatted in ISO format, see examples below or use time objects
        from the python standard library `datetime`.
    upper: str or datetime.time
        Latest possible time to be generated from this distribution.
        String formatted in ISO format, see examples below or use time objects
        from the python standard library `datetime`.
    precision: str
        Most precise measure of time that the distribution will take into account.
        For example, if precision == "seconds", then the microseconds will always be
        set to 0. Possible values: ["microseconds", "seconds", "minutes", "hours"].

    Examples
    --------
    >>> TimeUniformDistribution(lower="10:39:12", upper="10:39:45", precision="seconds")
    """

    def fromisoformat(self, dt_obj: str) -> dt.time:
        return dt.time.fromisoformat(dt_obj)

    @classmethod
    def default_distribution(cls):
        return cls("10:39:36", "18:39:36", precision="seconds")

    def draw(self):
        dt_lower = dt.datetime.combine(dt.datetime.today(), self.lower)
        dt_upper = dt.datetime.combine(dt.datetime.today(), self.upper)
        delta = dt_upper-dt_lower + self.minimum_delta
        return self.round((random()*delta + dt_lower).time())

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "string"},
            "upper": {"type": "string"},
            "precision": {"type": "string"},
        }


@metadist(name="core.uniform", var_type="date")
class DateUniformDistribution(BaseDTUniformDistribution):
    """Uniform date distribution.

    Date objects will be uniformly distributed between a start and end date.

    Parameters
    ----------
    lower: str or datetime.date
        Earliest possible date to be generated from this distribution.
        String formatted in ISO format, see examples below or use date objects
        from the python standard library `datetime`.
    upper: str or datetime.date
        Latest possible date to be generated from this distribution.
        String formatted in ISO format, see examples below or use date objects
        from the python standard library `datetime`.

    Examples
    --------
    >>> DateUniformDistribution(lower="10:39:12", upper="10:39:45", precision="seconds")
    """

    precision_possibilities = ["days"]

    def __init__(self, lower: Any, upper: Any):
        super().__init__(lower, upper, precision="days")

    def fromisoformat(self, dt_obj: str) -> dt.date:
        return dt.date.fromisoformat(dt_obj)

    def round(self, time_obj):
        return time_obj

    def _param_dict(self) -> Dict:
        date_dict = BaseDTUniformDistribution._param_dict(self)
        del date_dict["precision"]
        return date_dict

    @classmethod
    def default_distribution(cls):
        return cls("1903-07-15", "1940-07-16")

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "string"},
            "upper": {"type": "string"},
        }


class BaseDTUniformFitter(BaseFitter):
    """Base class for date/time/datetime uniform distributions."""

    precision_possibilities = ["microseconds", "seconds", "minutes", "hours", "days"]

    @classmethod
    def _get_precision(cls, values):
        cur_precision = 0
        for precision in cls.precision_possibilities[:-1]:
            if not np.all([getattr(d, precision[:-1]) == 0 for d in values]):
                break
            cur_precision += 1
        return cls.precision_possibilities[cur_precision]


@metafit(distribution=TimeUniformDistribution, var_type="time")
class TimeUniformFitter(BaseDTUniformFitter):
    """Fitter for time uniform distribution."""

    # precision_possibilities = ["microseconds", "seconds", "minutes", "hours", "days"]
    def _fit(self, values):
        return TimeUniformDistribution(values.min(), values.max(), self._get_precision(values))

@metafit(distribution=DateTimeUniformDistribution, var_type="datetime")
class DateTimeUniformFitter(BaseDTUniformFitter):
    """Fitter for datetime uniform distribution."""

    def _fit(self, values):
        return DateTimeUniformDistribution(values.min(), values.max(), self._get_precision(values))


@metafit(distribution=DateUniformDistribution, var_type="date")
class DateUniformFitter(BaseDTUniformFitter):
    """Fitter for date uniform distribution."""

    precision_possibilities = ["days"]
    def _fit(self, values):
        return DateUniformDistribution(values.min(), values.max())
