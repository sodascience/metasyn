"""Module for testing discrete distributions."""
from math import fabs

import numpy as np
import pandas as pd
import polars as pl
from pytest import mark
from scipy import stats

from metasyn.distribution.normal import (
    DiscreteNormalFitter,
    DiscreteTruncatedNormalFitter,
)
from metasyn.distribution.poisson import PoissonFitter
from metasyn.distribution.uniform import DiscreteUniformFitter
from metasyn.distribution.uniquekey import UniqueKeyDistribution, UniqueKeyFitter
from metasyn.privacy import BasicPrivacy


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
    """Test discrete uniform distribution."""
    series = series_type(data)
    dist = DiscreteUniformFitter(BasicPrivacy()).fit(series)
    assert dist.lower == series.min() and dist.upper == series.max()+1
    drawn_values = set([dist.draw() for _ in range(1000)])
    if dist.upper - dist.lower < 5:
        assert len(drawn_values) == len(series)
    drawn_values = np.array(list(drawn_values))
    assert np.isclose(dist.information_criterion(drawn_values),
                      np.log(len(drawn_values))*2+2*len(drawn_values)*(np.log(dist.upper-dist.lower)))


@mark.parametrize(
    "data,better_than_uniform,consecutive",
    [
        ([1, 2, 3, 4, 5], True, True),
        ([5, 4, 3, 2, 1], True, False),
        ([2, 4, 5, 7, 10, 6], True, False),
        ([-3, 1, -5, 3, -2, 0], True, False),
        ([-129384, 2198384, 293, 1293840], False, False),
        ([1, 1, 2, 2, 3, 3], False, False)
    ]
)
@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_integer_key(data, better_than_uniform, consecutive, series_type):
    """Test unique key distribution (included increasing id's)."""
    series = series_type(data)
    dist = UniqueKeyFitter(BasicPrivacy()).fit(series)
    unif_dist = DiscreteUniformFitter(BasicPrivacy()).fit(series)
    assert dist.lower == series.min()
    assert dist.consecutive == consecutive
    assert better_than_uniform == (dist.information_criterion(series)
                                   < unif_dist.information_criterion(series))
    assert isinstance(dist, UniqueKeyDistribution)

    drawn_values = np.array([dist.draw() for _ in range(100)])
    if consecutive:
        assert np.all(drawn_values == np.arange(dist.lower, dist.lower+100))


def test_unique_key_ic_zero():
    """Check behavior of BIC with zero values."""
    dist = UniqueKeyDistribution.default_distribution()
    assert dist.information_criterion([]) == 4

@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_poisson(series_type):
    """Test Poisson distribution."""
    series = series_type(stats.poisson(mu=10).rvs(1000))
    dist = PoissonFitter(BasicPrivacy()).fit(series)
    dist_unif = DiscreteUniformFitter(BasicPrivacy()).fit(series)
    assert fabs(dist.rate - 10) < 1
    assert dist.information_criterion(series) < dist_unif.information_criterion(series)

@mark.parametrize(
    "lower,upper,mean,sd",
    [
        (-1, 1, 0, 0.5),
        (-10, -8, 0, 5),
        (-5, 3, 2, 2),
    ]
)
def test_trunc_normal(lower, upper, mean, sd):
    """Test discrete version of truncated normal distribution."""
    a, b = (lower-mean)/sd, (upper-mean)/sd
    values = pl.Series(stats.truncnorm(a=a, b=b, loc=mean, scale=sd).rvs(5000)).cast(pl.Int64)
    dist = DiscreteTruncatedNormalFitter(BasicPrivacy()).fit(values)
    dist_uniform = DiscreteUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert isinstance(dist.draw(), int)


@mark.parametrize(
    "mean,sd",
    [
        (0, 1),
        (-10, 2),
        (100, 200),
    ]
)
def test_normal(mean, sd):
    """Test for discrete version of normal/Gaussian distribution."""
    values = pl.Series(stats.norm(loc=mean, scale=sd).rvs(1000)).cast(pl.Int64)
    dist = DiscreteNormalFitter(BasicPrivacy()).fit(values)
    dist_uniform = DiscreteUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert (dist.mean - mean)/sd < 0.5
    assert (dist.sd - sd)/sd < 0.5
    assert isinstance(dist.draw(), int)
