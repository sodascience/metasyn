from pytest import mark, raises

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.continuous import UniformDistribution,\
    NormalDistribution
from metasyn.distribution.faker import FakerDistribution
from metasyn.distribution.regex import RegexDistribution
from metasyn.distribution.discrete import DiscreteUniformDistribution
from metasyn.provider import DistributionProviderList


@mark.parametrize(
    "dist_str,dist",
    [
        ("multinoulli", MultinoulliDistribution),
        ("uniform", UniformDistribution),
        ("normal", NormalDistribution),
        ("faker", FakerDistribution),
        ("regex", RegexDistribution),
        ("discrete_uniform", DiscreteUniformDistribution)
    ]
)
def test_util(dist_str, dist):
    provider_list = DistributionProviderList("builtin")
    dist_class = provider_list.find_distribution(dist_str)
    assert dist == dist_class
    if "faker" in dist_str:
        with raises(ValueError):
            provider_list.find_distribution("this is not a distribution")
    new_class = provider_list.find_distribution(dist_class.__name__)
    assert new_class == dist_class
