from pytest import mark

from metasynth.provider import get_distribution_provider
from metasynth.testutils import check_distribution, check_distribution_provider


def test_builtin_provider():
    check_distribution_provider("builtin")


@mark.parametrize(
    "distribution", get_distribution_provider("builtin").distributions
)
def test_dist_validation(distribution):
    check_distribution(distribution, privacy="none",
                       provenance="builtin")
