"""Testing module for continuous distributions."""
import numpy as np
from pytest import mark
from scipy import stats

from metasyn.distribution.exponential import ExponentialFitter
from metasyn.distribution.normal import (
    LogNormalFitter,
    ContinuousNormalFitter,
    TruncatedNormalFitter,
)
from metasyn.distribution.uniform import ContinuousUniformFitter
from metasyn.privacy import BasicPrivacy


@mark.parametrize(
    "lower,upper",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_uniform(lower, upper):
    """Test the continuous uniform distribution."""
    scale = upper-lower
    values = stats.uniform(loc=lower, scale=scale).rvs(100)
    dist = ContinuousUniformFitter(BasicPrivacy()).fit(values)
    assert dist.lower <= values.min()
    assert dist.upper >= values.max()
    assert dist.information_criterion(values) < 2*np.log(len(values))-200*np.log((upper-lower)**-1)
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "mean,sd",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_normal(mean, sd):
    """Test the normal/Gaussian distribution."""
    values = stats.norm(loc=mean, scale=sd).rvs(1000)
    dist = ContinuousNormalFitter(BasicPrivacy()).fit(values)
    dist_uniform = ContinuousUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert (dist.mean - mean)/sd < 0.5
    assert (dist.sd - sd)/sd < 0.5
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "mean,sd",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_log_normal(mean, sd):
    """Test the log normal distribution."""
    values = stats.lognorm(s=sd, loc=0, scale=np.exp(mean)).rvs(1000)
    dist = LogNormalFitter(BasicPrivacy()).fit(values)
    dist_uniform = ContinuousUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert (dist.mean-mean)/sd < 1
    assert (dist.sd-sd)/sd < 1
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "lower,upper,mean,sd",
    [
        (-0.5, 0.5, 0, 0.5),
        (-10, -8, 0, 5),
        (-5, 3, 2, 2),
        (-0.01, 0.01, 0.01, 0.01),
    ]
)
def test_trunc_normal(lower, upper, mean, sd):
    """Test the truncated normal distribution."""
    a, b = (lower-mean)/sd, (upper-mean)/sd
    values = stats.truncnorm(a=a, b=b, loc=mean, scale=sd).rvs(5000)
    dist = TruncatedNormalFitter(BasicPrivacy()).fit(values)
    dist_uniform = ContinuousUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "rate",
    [0.1, 10, 100, 234.1234]
)
def test_exponential(rate):
    """Test the exponential distribution."""
    values = stats.expon(loc=0, scale=1/rate).rvs(5000)
    dist = ExponentialFitter(BasicPrivacy()).fit(values)
    dist_uniform = ContinuousUniformFitter(BasicPrivacy()).fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert isinstance(dist.draw(), float)
    assert (dist.rate - rate)/rate < 0.1
