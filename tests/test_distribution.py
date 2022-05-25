from pytest import mark, raises

from metasynth.distribution.util import get_dist_class
from metasynth.distribution.categorical import CatFreqDistribution
from metasynth.distribution.continuous import UniformDistribution,\
    NormalDistribution
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.string import RegexDistribution,\
    StringFreqDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution


@mark.parametrize(
    "dist_str,dist",
    [
        ("cat_freq", CatFreqDistribution),
        ("uniform", UniformDistribution),
        ("normal", NormalDistribution),
        ("gaussian", NormalDistribution),
        ("faker.city", FakerDistribution),
        ("faker.city.nl_NL", FakerDistribution),
        ("regex", RegexDistribution),
        ("char_freq", StringFreqDistribution),
        ("discrete_uniform", DiscreteUniformDistribution)
    ]
)
def test_util(dist_str, dist):
    dist_class, _ = get_dist_class(dist_str)
    assert dist == dist_class
    if dist_str.startswith("faker"):
        with raises(ValueError):
            get_dist_class(dist_class.__name__)
        with raises(ValueError):
            get_dist_class("this is not a distribution")
    else:
        new_class, _ = get_dist_class(dist_class.__name__)
        assert new_class == dist_class
