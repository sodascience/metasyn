import datetime as dt

from pytest import mark

from metasynth.distribution.datetime import UniformDateDistribution, UniformDateTimeDistribution
from metasynth.distribution.datetime import UniformTimeDistribution

all_precision = ["microseconds", "seconds", "minutes", "hours"]
begin_time = ["10", ""]


@mark.parametrize(
    "begin_time,end_time,precision", [
        ("10:23:45.293852", "16:23:45.293852", "microseconds"),
        ("10:23:45", "16:23:45", "seconds"),
        ("10:23:00", "16:23:00", "minutes"),
        ("10:00:00", "16:00:00", "hours"),
    ]
)
def test_time(begin_time, end_time, precision):
    begin_iso, end_iso = dt.time.fromisoformat(begin_time), dt.time.fromisoformat(end_time)
    dist = UniformTimeDistribution(begin_time, end_time, precision)
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.time)
        assert new_time >= begin_iso
        assert new_time <= end_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0


def test_date():
    begin_date, end_date = "2020-07-09", "2023-08-19"
    begin_iso, end_iso = dt.date.fromisoformat(begin_date), dt.date.fromisoformat(end_date)

    dist = UniformDateDistribution(begin_date, end_date)
    for _ in range(100):
        new_date = dist.draw()
        assert isinstance(new_date, dt.date)
        assert new_date >= begin_iso
        assert new_date <= end_iso


@mark.parametrize(
    "begin_time,end_time,precision", [
        ("2020-07-09 10:23:45.293852", "2020-08-09 16:23:45.293852", "microseconds"),
        ("2020-07-09 10:23:45", "2020-08-09 16:23:45", "seconds"),
        ("2020-07-09 10:23:00", "2020-08-09 16:23:00", "minutes"),
        ("2020-07-09 10:00:00", "2020-08-09 16:00:00", "hours"),
    ]
)
def test_datetime(begin_time, end_time, precision):
    begin_iso, end_iso = dt.datetime.fromisoformat(begin_time), dt.datetime.fromisoformat(end_time)
    dist = UniformDateTimeDistribution(begin_time, end_time, precision)
    for _ in range(100):
        new_time = dist.draw()
        assert isinstance(new_time, dt.datetime)
        assert new_time >= begin_iso
        assert new_time <= end_iso
        for prec in all_precision:
            if prec == precision:
                break
            assert getattr(new_time, prec[:-1]) == 0
