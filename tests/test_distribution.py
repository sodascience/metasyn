from pytest import mark, raises

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.continuous import NormalDistribution, UniformDistribution
from metasyn.distribution.discrete import DiscreteNormalDistribution, DiscreteUniformDistribution
from metasyn.distribution.string import (
    FakerDistribution,
    RegexDistribution,
    UniqueRegexDistribution,
)
from metasyn.provider import DistributionProviderList


@mark.parametrize(
    "dist_str,var_type,is_unique,dist",
    [
        ("multinoulli", "string", False, MultinoulliDistribution),
        ("multinoulli", "discrete", False, MultinoulliDistribution),
        ("uniform", "continuous", False, UniformDistribution),
        ("uniform", "discrete", False, DiscreteUniformDistribution),
        ("normal", "continuous", False, NormalDistribution),
        ("normal", "discrete", False, DiscreteNormalDistribution),
        ("faker", "string", False, FakerDistribution),
        ("regex", "string", False, RegexDistribution),
        ("regex", "string", True, UniqueRegexDistribution),
    ]
)
def test_util(dist_str, var_type, is_unique, dist):
    provider_list = DistributionProviderList("builtin")
    dist_class = provider_list.find_distribution(dist_str, var_type=var_type, unique=is_unique)
    assert dist == dist_class
    if "faker" in dist_str:
        with raises(ValueError):
            provider_list.find_distribution("this is not a distribution", "string")
    new_class = provider_list.find_distribution(dist_class.__name__, var_type=var_type, unique=is_unique)
    assert new_class == dist_class
