"""Module implementing distributions for date and time types."""

import datetime as dt
from abc import abstractmethod
from random import random
from typing import Any, Dict

import numpy as np

from metasyn.distribution.base import BaseDistribution, metadist


def convert_numpy_datetime(time_obj: np.datetime64) -> dt.datetime:
    """Convert numpy datetime to python stdlib datetime.

    Parameters
    ----------
    time_obj:
        datetime to be converted.

    Returns
    -------
    datetime.datetime:
        Converted datetime.
    """
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (time_obj - unix_epoch) / one_second
    return dt.datetime.utcfromtimestamp(float(seconds_since_epoch))


class BaseUniformDistribution(BaseDistribution):
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

    @classmethod
    def _fit(cls, values):
        return cls(values.min(), values.max(), cls._get_precision(values))

    @classmethod
    def _get_precision(cls, values):
        cur_precision = 0
        for precision in cls.precision_possibilities[:-1]:
            if not np.all([getattr(d, precision[:-1]) == 0 for d in values]):
                break
            cur_precision += 1
        return cls.precision_possibilities[cur_precision]

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

    def information_criterion(self, values):
        return 0.0

    def _param_dict(self):
        return {
            "lower": self.lower.isoformat(),
            "upper": self.upper.isoformat(),
            "precision": self.precision,
        }


@metadist(implements="core.uniform", var_type="datetime")
class DateTimeUniformDistribution(BaseUniformDistribution):
    """Uniform DateTime distribution."""

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


@metadist(implements="core.uniform", var_type="time")
class TimeUniformDistribution(BaseUniformDistribution):
    """Uniform time distribution."""

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


@metadist(implements="core.uniform", var_type="date")
class DateUniformDistribution(BaseUniformDistribution):
    """Uniform date distribution."""

    precision_possibilities = ["days"]

    def __init__(self, lower: Any, upper: Any):
        super().__init__(lower, upper, precision="days")

    def fromisoformat(self, dt_obj: str) -> dt.date:
        return dt.date.fromisoformat(dt_obj)

    def round(self, time_obj):
        return time_obj

    def _param_dict(self) -> Dict:
        date_dict = BaseUniformDistribution._param_dict(self)
        del date_dict["precision"]
        return date_dict

    @classmethod
    def default_distribution(cls):
        return cls("1903-07-15", "1940-07-16")

    @classmethod
    def _fit(cls, values):
        return cls(values.min(), values.max())

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "string"},
            "upper": {"type": "string"},
        }
