from pytest import mark, raises

from metasyn.distribution import UniformDistribution
from metasyn.privacy import BasicPrivacy
from metasyn.provider import get_distribution_provider
from metasyn.testutils import check_distribution, check_distribution_provider


def test_builtin_provider():
    check_distribution_provider("builtin")


@mark.parametrize(
    "distribution", get_distribution_provider("builtin").distributions
)
def test_dist_validation(distribution):
    check_distribution(distribution, privacy=BasicPrivacy(),
                       provenance="builtin")


class Distribution(UniformDistribution):
    def schema():
        return "[{"


def test_schema_val():
    with raises(ValueError):
        check_distribution(Distribution, "none", "builtin")
