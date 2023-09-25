import datetime as dt

from pytest import mark
import pandas as pd
import polars as pl

from metasyn.distribution.datetime import UniformDateDistribution, UniformDateTimeDistribution
from metasyn.distribution.datetime import UniformTimeDistribution

all_precision = ["microseconds", "seconds", "minutes", "hours"]
start = ["10", ""]


@mark.parametrize(
    "start,end,precision", [
        ("10:23:45.293852", "16:23:45.293852", "microseconds"),
        ("10:23:45", "16:23:45", "seconds"),
        ("10:23:00", "16:23:00", "minutes"),
        ("10:00:00", "16:00:00", "hours"),
    ]
)
@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_time(start, end, precision, series_type):
    begin_iso, end_iso = dt.time.fromisoformat(start), dt.time.fromisoformat(end)
    dist = UniformTimeDistribution(start, end, precision)
    all_times = []
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.time)
        assert new_time >= begin_iso
        assert new_time <= end_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0
        all_times.append(new_time)
    series = series_type(all_times)
    new_dist = UniformTimeDistribution.fit(series)
    assert new_dist.start >= dist.start
    assert new_dist.end <= dist.end
    assert new_dist.precision == dist.precision


@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_date(series_type):
    begin_date, end_date = "2020-07-09", "2023-08-19"
    begin_iso, end_iso = dt.date.fromisoformat(begin_date), dt.date.fromisoformat(end_date)

    dist = UniformDateDistribution(begin_date, end_date)
    all_dates = []
    for _ in range(100):
        new_date = dist.draw()
        assert isinstance(new_date, dt.date)
        assert new_date >= begin_iso
        assert new_date <= end_iso
        all_dates.append(new_date)
    series = series_type(all_dates)
    new_dist = UniformDateDistribution.fit(series)
    assert new_dist.start >= dist.start
    assert new_dist.end <= dist.end
    assert new_dist.precision == dist.precision


@mark.parametrize(
    "start,end,precision", [
        ("2020-07-09 10:23:45.293852", "2020-08-09 16:23:45.293852", "microseconds"),
        ("2020-07-09 10:23:45", "2020-08-09 16:23:45", "seconds"),
        ("2020-07-09 10:23:00", "2020-08-09 16:23:00", "minutes"),
        ("2020-07-09 10:00:00", "2020-08-09 16:00:00", "hours"),
    ]
)
@mark.parametrize("series_type", [pl.Series, pd.Series])
def test_datetime(start, end, precision, series_type):
    begin_iso, end_iso = dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end)
    dist = UniformDateTimeDistribution(start, end, precision)
    all_datetimes = []
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.datetime)
        assert new_time >= begin_iso
        assert new_time <= end_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0
        all_datetimes.append(new_time)

    series = series_type(all_datetimes)
    new_dist = UniformDateTimeDistribution.fit(series)
    assert new_dist.start >= dist.start
    assert new_dist.end <= dist.end
    assert new_dist.precision == dist.precision
