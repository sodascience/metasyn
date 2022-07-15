"""Distributions for date and time types."""

from abc import abstractmethod
from random import random
import datetime as dt
from typing import Dict, Any

from metasynth.distribution.base import DateTimeDistribution, TimeDistribution
from metasynth.distribution.base import ScipyDistribution, DateDistribution


class BaseUniformDistribution(ScipyDistribution):
    """Base class for all time/date/datetime uniform distributions."""
    def __init__(self, begin_time: Any, end_time: Any):
        if isinstance(begin_time, str):
            begin_time = self.fromisoformat(begin_time)
        if isinstance(end_time, str):
            end_time = self.fromisoformat(end_time)
        self.par = {
            "begin_time": begin_time,
            "end_time": end_time
        }

    @classmethod
    def _fit(cls, values):
        return cls(values.min(), values.max())

    def draw(self) -> dt.datetime:
        delta = self.end_time-self.begin_time + self.minimum_delta
        return random()*delta + self.begin_time

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "parameters": {
                "begin_time": self.begin_time.isoformat(),
                "end_time": self.end_time.isoformat(),
            }
        }

    @abstractmethod
    def fromisoformat(self, dt_obj: str):
        """Convert string to iso format."""

    @property
    @abstractmethod
    def minimum_delta(self) -> dt.timedelta:
        """Get the minimum time delta"""

    def information_criterion(self, values):
        return 0.0


class UniformDateTimeDistribution(DateTimeDistribution, BaseUniformDistribution):
    """Uniform DateTime distribution."""
    aliases = ["UniformDateTimeDistribution", "datetime_uniform"]

    def fromisoformat(self, dt_obj: str) -> dt.datetime:
        return dt.datetime.fromisoformat(dt_obj)

    @property
    def minimum_delta(self) -> dt.timedelta:
        return dt.timedelta(microseconds=1)

    @classmethod
    def _example_distribution(cls):
        return cls("2022-07-15T10:39:36.130261", "2022-08-15T10:39:36.130261")


class UniformTimeDistribution(TimeDistribution, BaseUniformDistribution):
    """Uniform time distribution."""
    aliases = ["UniformTimeDistribution", "time_uniform"]

    def fromisoformat(self, dt_obj: str) -> dt.time:
        return dt.time.fromisoformat(dt_obj)

    @property
    def minimum_delta(self) -> dt.timedelta:
        return dt.timedelta(microseconds=1)

    @classmethod
    def _example_distribution(cls):
        return cls("10:39:36.130261", "18:39:36.130261")


class UniformDateDistribution(DateDistribution, BaseUniformDistribution):
    """Uniform date distribution."""
    aliases = ["UniformDateDistribution", "date_uniform"]

    def fromisoformat(self, dt_obj: str) -> dt.date:
        return dt.date.fromisoformat(dt_obj)

    @property
    def minimum_delta(self) -> dt.timedelta:
        return dt.timedelta(days=1)

    @classmethod
    def _example_distribution(cls):
        return cls("2022-07-15", "2022-07-16")
