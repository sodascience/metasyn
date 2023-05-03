from pytest import mark, raises

from metasynth.distribution.categorical import MultinoulliDistribution
from metasynth.distribution.continuous import UniformDistribution,\
    NormalDistribution
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.regex import RegexDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution
from metasynth.provider import DistributionProviderList


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
    dist_class = provider_list._find_distribution(dist_str)
    assert dist == dist_class
    if dist_str.startswith("faker"):
        with raises(ValueError):
            provider_list._find_distribution("this is not a distribution")
    new_class = provider_list._find_distribution(dist_class.__name__)
    assert new_class == dist_class
