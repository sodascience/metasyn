import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import poisson

from metasyn.distribution.discrete import UniqueKeyDistribution, DiscreteUniformDistribution,\
    PoissonDistribution
from pytest import mark
from math import fabs


@mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5],
        [-399, 12, 1, 492],
        [np.random.randint(0, 1000) for _ in range(1000)],
    ]
)
@mark.parametrize(
    "series_type", [pd.Series, pl.Series]
)
def test_uniform(data, series_type):
    series = series_type(data)
    dist = DiscreteUniformDistribution.fit(series)
    assert dist.low == series.min() and dist.high == series.max()+1
    drawn_values = set([dist.draw() for _ in range(1000)])
    if dist.high - dist.low < 5:
        assert len(drawn_values) == len(series)
    drawn_values = np.array(list(drawn_values))
    assert np.isclose(dist.information_criterion(drawn_values),
                      4+2*len(drawn_values)*(np.log(dist.high-dist.low)))


@mark.parametrize(
    "data,better_than_uniform,consecutive",
    [
        ([1, 2, 3, 4, 5], True, 1),
        ([5, 4, 3, 2, 1], True, 0),
        ([2, 4, 5, 7, 10, 6], True, 0),
        ([-3, 1, -5, 3, -2, 0], True, 0),
        ([-129384, 2198384, 293, 1293840], False, 0),
        ([1, 1, 2, 2, 3, 3], False, 0)
    ]
)
@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_integer_key(data, better_than_uniform, consecutive, series_type):
    series = series_type(data)
    dist = UniqueKeyDistribution.fit(series)
    unif_dist = DiscreteUniformDistribution.fit(series)
    assert dist.low == series.min()
    assert dist.consecutive == consecutive
    assert better_than_uniform == (dist.information_criterion(series) < unif_dist.information_criterion(series))
    assert isinstance(dist, UniqueKeyDistribution)

    drawn_values = np.array([dist.draw() for _ in range(100)])
    if consecutive:
        assert np.all(drawn_values == np.arange(dist.low, dist.low+100))


@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_poisson(series_type):
    series = series_type(poisson(mu=10).rvs(1000))
    dist = PoissonDistribution.fit(series)
    dist_unif = DiscreteUniformDistribution.fit(series)
    assert fabs(dist.mu - 10) < 1
    assert dist.information_criterion(series) < dist_unif.information_criterion(series)
