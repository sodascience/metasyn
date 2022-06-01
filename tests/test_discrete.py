import pandas as pd
import numpy as np
from scipy.stats import poisson

from metasynth.distribution.discrete import UniqueKeyDistribution, DiscreteUniformDistribution,\
    PoissonDistribution
from pytest import mark
from math import fabs


@mark.parametrize(
    "series",
    [
        pd.Series([1, 2, 3, 4, 5]),
        pd.Series([-399, 12, 1, 492]),
        pd.Series([np.random.randint(0, 1000) for _ in range(1000)]),
    ]
)
def test_uniform(series):
    dist = DiscreteUniformDistribution.fit(series)
    assert dist.low == np.min(series) and dist.high == np.max(series)+1
    drawn_values = set([dist.draw() for _ in range(1000)])
    if dist.high - dist.low < 5:
        assert len(drawn_values) == len(series)
    drawn_values = np.array(list(drawn_values))
    assert np.isclose(dist.information_criterion(drawn_values),
                      4+2*len(drawn_values)*(np.log(dist.high-dist.low)))


@mark.parametrize(
    "series,better_than_uniform,consecutive",
    [
        (pd.Series([1, 2, 3, 4, 5]), True, 1),
        (pd.Series([5, 4, 3, 2, 1]), True, 0),
        (pd.Series([2, 4, 5, 7, 10, 6]), True, 0),
        (pd.Series([-3, 1, -5, 3, -2, 0]), True, 0),
        (pd.Series([-129384, 2198384, 293, 1293840]), False, 0),
        (pd.Series([1, 1, 2, 2, 3, 3]), False, 0)
    ]
)
def test_integer_key(series, better_than_uniform, consecutive):
    dist = UniqueKeyDistribution.fit(series)
    unif_dist = DiscreteUniformDistribution.fit(series)
    assert dist.low == np.min(series)
    assert dist.consecutive == consecutive
    assert better_than_uniform == (dist.information_criterion(series) < unif_dist.information_criterion(series))
    assert isinstance(dist, UniqueKeyDistribution)

    drawn_values = np.array([dist.draw() for _ in range(100)])
    if consecutive:
        assert np.all(drawn_values == np.arange(dist.low, dist.low+100))


def test_poisson():
    series = pd.Series(poisson(mu=10).rvs(1000))
    dist = PoissonDistribution.fit(series)
    dist_unif = DiscreteUniformDistribution.fit(series)
    assert fabs(dist.mu - 10) < 1
    assert dist.information_criterion(series) < dist_unif.information_criterion(series)
