"""Module to test the provider and provider list classes."""

import pytest
from pytest import mark, raises

from metasyn.distribution.base import BaseDistribution
from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.faker import FakerDistribution
from metasyn.distribution.normal import ContinuousNormalDistribution, DiscreteNormalDistribution
from metasyn.distribution.regex import RegexDistribution, UniqueRegexDistribution
from metasyn.distribution.uniform import ContinuousUniformDistribution, DiscreteUniformDistribution
from metasyn.provider import BuiltinDistributionProvider, DistributionProviderList
from metasyn.varspec import DistributionSpec, VarSpec


@mark.parametrize("input", ["builtin", "fake-name", BuiltinDistributionProvider,
                            BuiltinDistributionProvider()])
def test_plist_init(input):
    """Test whether the initialization works correctly."""
    if input == "fake-name":
        with pytest.raises(ValueError):
            DistributionProviderList(input)
    else:
        plist = DistributionProviderList(input)
        assert issubclass(plist.find_distribution("core.regex", var_type="string"),
                          BaseDistribution)


# @metadist(version="1.0")
# class UniformTest1(ContinuousUniformDistribution):
#     """Version 1.0 for legacy tests."""


# @metadist(version="1.1")
# class UniformTest11(UniformDistribution):
#     """Version 1.1 for legacy tests."""


# @metadist(version="2.0")
# class UniformTest2(UniformDistribution):
#     """Version 2.0 for legacy tests."""


# class CheckProvider(BuiltinDistributionProvider):
#     """Provider for legacy tests."""

#     distributions = [UniformTest2]
#     legacy_distributions = [UniformTest1, UniformTest11]


# class LegacyOnly(BuiltinDistributionProvider):
#     """Provider with deprecated uniform distributions."""

#     distributions = [MultinoulliDistribution]
#     legacy_distributions = [UniformTest1, UniformTest11, UniformTest2]


# def test_legacy():
#     """Test whether the legacy system/distributions work."""
#     plist = DistributionProviderList(CheckProvider)
#     assert issubclass(plist.find_distribution("core.uniform", var_type="continuous"), UniformTest2)
#     with pytest.warns():  # version 1.1 is deprecated
#         assert issubclass(
#             plist.find_distribution(
#                 "core.uniform", var_type="continuous", version="1.1"), UniformTest11)
#     with pytest.warns():  # version 1.0 is deprecated
#         assert issubclass(
#             plist.find_distribution("core.uniform", var_type="continuous", version="1.0"),
#             UniformTest1)
#     with pytest.warns():  # version 1.2 does not exist, so 1.1 should be chosen.
#         assert issubclass(
#             plist.find_distribution("core.uniform", var_type="continuous", version="1.2"),
#             UniformTest11)
#     with pytest.raises(ValueError):  # No version 0.x available
#         plist.find_distribution("core.uniform", var_type="continuous", version="0.9")
#     with pytest.warns():  # No version 2.1 available, but 2.0 is.
#         assert issubclass(
#             plist.find_distribution("core.uniform", var_type="continuous", version="2.1"),
#             UniformTest2)

#     plist = DistributionProviderList(LegacyOnly)
#     with pytest.warns():  # core.uniform is deprecated.
#         assert issubclass(
#             plist.find_distribution("core.uniform", var_type="continuous"),
#             UniformTest2)


@mark.parametrize(
    "dist_str,var_type,is_unique,dist",
    [
        ("multinoulli", "string", False, MultinoulliDistribution),
        ("multinoulli", "discrete", False, MultinoulliDistribution),
        ("uniform", "continuous", False, ContinuousUniformDistribution),
        ("uniform", "discrete", False, DiscreteUniformDistribution),
        ("normal", "continuous", False, ContinuousNormalDistribution),
        ("normal", "discrete", False, DiscreteNormalDistribution),
        ("faker", "string", False, FakerDistribution),
        ("regex", "string", False, RegexDistribution),
        ("regex", "string", True, UniqueRegexDistribution),
    ]
)
def test_find_distribution(dist_str, var_type, is_unique, dist):
    """Test whether distributions can be found using the providers."""
    provider_list = DistributionProviderList("builtin")
    dist_class = provider_list.find_distribution(dist_str, var_type=var_type, unique=is_unique)
    assert dist == dist_class
    if "faker" in dist_str:
        with raises(ValueError):
            provider_list.find_distribution("this is not a distribution", "string")
    new_class = provider_list.find_distribution(dist_class.__name__, var_type=var_type,
                                                unique=is_unique)
    assert new_class == dist_class

def test_create_distribution():
    dist_spec = DistributionSpec("uniform", False, parameters={"lower": 10, "upper": 20})
    var_spec = VarSpec("test", dist_spec, var_type="continuous")
    provider_list = DistributionProviderList("builtin")

    assert isinstance(provider_list.create(var_spec), ContinuousUniformDistribution)

    # Error with missing parameters
    var_spec.dist_spec.parameters.pop("lower")
    with pytest.raises(ValueError):
        provider_list.create(var_spec)

    # Error with unknown parameters
    with pytest.raises(TypeError):
        var_spec.dist_spec.parameters["unknown"] = 1
        provider_list.create(var_spec)

    # Error when name is not given.
    with pytest.raises(ValueError):
        var_spec.dist_spec.name = None
        provider_list.create(var_spec)
