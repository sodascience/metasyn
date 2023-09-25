import numpy as np
from scipy import stats
from metasyn.distribution.continuous import UniformDistribution,\
    NormalDistribution, LogNormalDistribution, TruncatedNormalDistribution,\
    ExponentialDistribution
from pytest import mark


@mark.parametrize(
    "lower_bound,upper_bound",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_uniform(lower_bound, upper_bound):
    scale = upper_bound-lower_bound
    values = stats.uniform(loc=lower_bound, scale=scale).rvs(100)
    dist = UniformDistribution.fit(values)
    assert dist.min_val <= values.min()
    assert dist.max_val >= values.max()
    assert dist.information_criterion(values) < 4 - 200*np.log((upper_bound-lower_bound)**-1)
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "mean,std_dev",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_normal(mean, std_dev):
    values = stats.norm(loc=mean, scale=std_dev).rvs(1000)
    dist = NormalDistribution.fit(values)
    dist_uniform = UniformDistribution.fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert (dist.mean - mean)/std_dev < 0.5
    assert (dist.std_dev - std_dev)/std_dev < 0.5
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "mu,sigma",
    [
        (0, 1),
        (-10, 2),
        (0.01, 0.02),
    ]
)
def test_log_normal(mu, sigma):
    values = stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)).rvs(1000)
    dist = LogNormalDistribution.fit(values)
    dist_uniform = UniformDistribution.fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert (dist.mu-mu)/sigma < 1
    assert (dist.sigma-sigma)/sigma < 1
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "lower_bound,upper_bound,mu,sigma",
    [
        (-0.5, 0.5, 0, 0.5),
        (-10, -8, 0, 5),
        (-5, 3, 2, 2),
        (-0.01, 0.01, 0.01, 0.01),
    ]
)
def test_trunc_normal(lower_bound, upper_bound, mu, sigma):
    a, b = (lower_bound-mu)/sigma, (upper_bound-mu)/sigma
    values = stats.truncnorm(a=a, b=b, loc=mu, scale=sigma).rvs(5000)
    dist = TruncatedNormalDistribution.fit(values)
    dist_uniform = UniformDistribution.fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert isinstance(dist.draw(), float)


@mark.parametrize(
    "rate",
    [0.1, 10, 100, 234.1234]
)
def test_exponential(rate):
    values = stats.expon(loc=0, scale=1/rate).rvs(5000)
    dist = ExponentialDistribution.fit(values)
    dist_uniform = UniformDistribution.fit(values)
    assert dist.information_criterion(values) < dist_uniform.information_criterion(values)
    assert isinstance(dist.draw(), float)
    assert (dist.rate - rate)/rate < 0.1
