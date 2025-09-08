"""Testing module for the distributions that are builtin to metasyn directly."""
from pytest import mark, raises

from metasyn.distribution.uniform import ContinuousUniformFitter, ContinuousUniformDistribution
from metasyn.distribution.base import metafit
from metasyn.privacy import BasicPrivacy
from metasyn.provider import get_distribution_provider
from metasyn.testutils import check_fitter, check_distribution_provider


def test_builtin_provider():
    """Test the builtin distribution provider with the metasyn test functionality."""
    check_distribution_provider("builtin")


@mark.parametrize(
    "fitter", get_distribution_provider("builtin").fitters
)
def test_dist_validation(fitter):
    """Check all distributions for correctness."""
    check_fitter(fitter, privacy=BasicPrivacy(), provenance="builtin")


class BadDistribution(ContinuousUniformDistribution):
    """Distribution that has a broken schema."""

    def schema():
        return "[{"

@metafit(distribution=BadDistribution)
class BadFitter(ContinuousUniformFitter):
    pass

def test_schema_val():
    """Check whether a distribution with a broken schema will trigger the error."""
    with raises(ValueError):
        check_fitter(BadFitter, "none", "builtin")
