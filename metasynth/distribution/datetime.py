"""Distributions for date and time types."""

from abc import abstractmethod
from random import random
import datetime as dt
from typing import Dict, Any

import numpy as np

from metasynth.distribution.base import DateTimeDistribution, TimeDistribution
from metasynth.distribution.base import ScipyDistribution, DateDistribution


class BaseUniformDistribution(ScipyDistribution):
    """Base class for all time/date/datetime uniform distributions."""

    precision_possibilities = ["microseconds", "seconds", "minutes", "hours", "days"]

    def __init__(self, start: Any, end: Any, precision: str="microseconds"):
        if isinstance(start, str):
            start = self.fromisoformat(start)
        if isinstance(end, str):
            end = self.fromisoformat(end)
        self.par = {
            "start": start,
            "end": end,
            "precision": precision,
        }

    @classmethod
    def _fit(cls, values):
        cur_precision = 0
        for precision in cls.precision_possibilities[:-1]:
            if not np.all([getattr(d, precision[:-1]) == 0 for d in values]):
                break
            cur_precision += 1
        return cls(values.min(), values.max(), cls.precision_possibilities[cur_precision])

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
        return time_obj

    def draw(self) -> dt.datetime:
        delta = self.end-self.start + self.minimum_delta
        new_time = random()*delta + self.start
        return self.round(new_time)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "parameters": {
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "precision": self.precision,
            }
        }

    @abstractmethod
    def fromisoformat(self, dt_obj: str):
        """Convert string to iso format."""

    @property
    def minimum_delta(self) -> dt.timedelta:
        """Get the minimum time delta."""
        return dt.timedelta(**{self.precision: 1})

    def information_criterion(self, values):
        return 0.0


class UniformDateTimeDistribution(DateTimeDistribution, BaseUniformDistribution):
    """Uniform DateTime distribution."""

    aliases = ["UniformDateTimeDistribution", "datetime_uniform"]

    def fromisoformat(self, dt_obj: str) -> dt.datetime:
        return dt.datetime.fromisoformat(dt_obj)

    @classmethod
    def _example_distribution(cls):
        return cls("2022-07-15T10:39:36", "2022-08-15T10:39:36")


class UniformTimeDistribution(TimeDistribution, BaseUniformDistribution):
    """Uniform time distribution."""

    aliases = ["UniformTimeDistribution", "time_uniform"]

    def fromisoformat(self, dt_obj: str) -> dt.time:
        return dt.time.fromisoformat(dt_obj)

    @classmethod
    def _example_distribution(cls):
        return cls("10:39:36", "18:39:36")

    def draw(self):
        dt_begin = dt.datetime.combine(dt.datetime.today(), self.start)
        dt_end = dt.datetime.combine(dt.datetime.today(), self.end)
        delta = dt_end-dt_begin + self.minimum_delta
        return self.round((random()*delta + dt_begin).time())


class UniformDateDistribution(DateDistribution, BaseUniformDistribution):
    """Uniform date distribution."""

    aliases = ["UniformDateDistribution", "date_uniform"]
    precision_possibilities = ["days"]

    def __init__(self, start: Any, end: Any):
        super().__init__(start, end, precision="days")

    def fromisoformat(self, dt_obj: str) -> dt.date:
        return dt.date.fromisoformat(dt_obj)

    def round(self, time_obj):
        return time_obj

    def to_dict(self) -> Dict:
        date_dict = BaseUniformDistribution.to_dict(self)
        date_dict["parameters"].pop("precision")
        return date_dict

    @classmethod
    def _example_distribution(cls):
        return cls("1903-07-15", "1940-07-16")

    @classmethod
    def _fit(cls, values):
        return cls(values.min(), values.max())
