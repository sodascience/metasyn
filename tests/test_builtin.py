from pytest import mark

from metasyn.provider import get_distribution_provider
from metasyn.testutils import check_distribution, check_distribution_provider
from metasyn.privacy import BasicPrivacy


def test_builtin_provider():
    check_distribution_provider("builtin")


@mark.parametrize(
    "distribution", get_distribution_provider("builtin").distributions
)
def test_dist_validation(distribution):
    check_distribution(distribution, privacy=BasicPrivacy(),
                       provenance="builtin")
