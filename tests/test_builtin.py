"""Testing module for the distributions that are builtin to metasyn directly."""
from pytest import mark, raises

from metasyn.distribution.base import metafit
from metasyn.distribution.uniform import ContinuousUniformDistribution, ContinuousUniformFitter
from metasyn.privacy import BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.testutils import check_distribution_registry, check_fitter


def test_builtin_registry():
    """Test the builtin distribution registry with the metasyn test functionality."""
    check_distribution_registry("builtin")


@mark.parametrize(
    "fitter", DistributionRegistry.parse("builtin").fitters
)
def test_dist_validation(fitter):
    """Check all distributions for correctness."""
    print(type(fitter), fitter)
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
