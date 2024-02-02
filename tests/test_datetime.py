import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
from pytest import mark

from metasyn.distribution.datetime import (
    DateTimeUniformDistribution,
    DateUniformDistribution,
    TimeUniformDistribution,
)

all_precision = ["microseconds", "seconds", "minutes", "hours"]
lower = ["10", ""]


@mark.parametrize(
    "lower,upper,precision", [
        ("10:23:45.293852", "16:23:45.293852", "microseconds"),
        ("10:23:45", "16:23:45", "seconds"),
        ("10:23:00", "16:23:00", "minutes"),
        ("10:00:00", "16:00:00", "hours"),
    ]
)
@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_time(lower, upper, precision, series_type):
    lower_iso, upper_iso = dt.time.fromisoformat(lower), dt.time.fromisoformat(upper)
    dist = TimeUniformDistribution(lower, upper, precision)
    all_times = []
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.time)
        assert new_time >= lower_iso
        assert new_time <= upper_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0
        all_times.append(new_time)
    series = series_type(all_times)
    new_dist = TimeUniformDistribution.fit(series)
    assert new_dist.lower >= dist.lower
    assert new_dist.upper <= dist.upper
    assert new_dist.precision == dist.precision


@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_date(series_type):
    lower_date, upper_date = "2020-07-09", "2023-08-19"
    lower_iso, upper_iso = dt.date.fromisoformat(lower_date), dt.date.fromisoformat(upper_date)

    dist = DateUniformDistribution(lower_date, upper_date)
    all_dates = []
    for _ in range(100):
        new_date = dist.draw()
        assert isinstance(new_date, dt.date)
        assert new_date >= lower_iso
        assert new_date <= upper_iso
        all_dates.append(new_date)
    series = series_type(all_dates)
    new_dist = DateUniformDistribution.fit(series)
    assert new_dist.lower >= dist.lower
    assert new_dist.upper <= dist.upper
    assert new_dist.precision == dist.precision


@mark.parametrize(
    "lower,upper,precision", [
        ("2020-07-09 10:23:45.293852", "2020-08-09 16:23:45.293852", "microseconds"),
        ("2020-07-09 10:23:45", "2020-08-09 16:23:45", "seconds"),
        ("2020-07-09 10:23:00", "2020-08-09 16:23:00", "minutes"),
        ("2020-07-09 10:00:00", "2020-08-09 16:00:00", "hours"),
        (np.datetime64('2005-02-25T03:30'), np.datetime64('2006-02-25T03:30'), "minutes")
    ]
)
@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_datetime(lower, upper, precision, series_type):
    if isinstance(lower, str):
        lower_iso, upper_iso = dt.datetime.fromisoformat(lower), dt.datetime.fromisoformat(upper)
    else:
        lower_iso = lower
        upper_iso = upper
    dist = DateTimeUniformDistribution(lower, upper, precision)
    all_datetimes = []
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.datetime)
        assert new_time >= lower_iso
        assert new_time <= upper_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0
        all_datetimes.append(new_time)

    series = series_type(all_datetimes)
    new_dist = DateTimeUniformDistribution.fit(series)
    assert new_dist.lower >= dist.lower
    assert new_dist.upper <= dist.upper
    assert new_dist.precision == dist.precision
