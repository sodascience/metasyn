"""Testing module for the distributions that are builtin to metasyn directly."""
from pytest import mark, raises

from metasyn.distribution import UniformDistribution
from metasyn.privacy import BasicPrivacy
from metasyn.provider import get_distribution_provider
from metasyn.testutils import check_distribution, check_distribution_provider


def test_builtin_provider():
    """Test the builtin distribution provider with the metasyn test functionality."""
    check_distribution_provider("builtin")


@mark.parametrize(
    "distribution", get_distribution_provider("builtin").distributions
)
def test_dist_validation(distribution):
    """Check all distributions for correctness."""
    check_distribution(distribution, privacy=BasicPrivacy(),
                       provenance="builtin")


class BadDistribution(UniformDistribution):
    """Distribution that has a broken schema."""

    def schema():
        return "[{"


def test_schema_val():
    """Check whether a distribution with a broken schema will trigger the error."""
    with raises(ValueError):
        check_distribution(BadDistribution, "none", "builtin")
