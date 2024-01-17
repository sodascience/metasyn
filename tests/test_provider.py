import pytest
from pytest import mark

from metasyn.distribution import MultinoulliDistribution, UniformDistribution
from metasyn.distribution.base import BaseDistribution, metadist
from metasyn.provider import BuiltinDistributionProvider, DistributionProviderList


@mark.parametrize("input", ["builtin", "fake-name", BuiltinDistributionProvider,
                            BuiltinDistributionProvider()])
def test_plist_init(input):
    if input == "fake-name":
        with pytest.raises(ValueError):
            DistributionProviderList(input)
    else:
        plist = DistributionProviderList(input)
        assert issubclass(plist.find_distribution("core.regex", var_type="string"), BaseDistribution)


@metadist(version="1.0")
class UniformTest1(UniformDistribution):
    pass


@metadist(version="1.1")
class UniformTest11(UniformDistribution):
    pass


@metadist(version="2.0")
class UniformTest2(UniformDistribution):
    pass


class CheckProvider(BuiltinDistributionProvider):
    distributions = [UniformTest2]
    legacy_distributions = [UniformTest1, UniformTest11]


class LegacyOnly(BuiltinDistributionProvider):
    distributions = [MultinoulliDistribution]
    legacy_distributions = [UniformTest1, UniformTest11, UniformTest2]


def test_legacy():
    plist = DistributionProviderList(CheckProvider)
    assert issubclass(plist.find_distribution("core.uniform", var_type="continuous"), UniformTest2)
    with pytest.warns():
        assert issubclass(plist.find_distribution("core.uniform", var_type="continuous", version="1.1"), UniformTest11)
    with pytest.warns():
        assert issubclass(plist.find_distribution("core.uniform", var_type="continuous", version="1.0"), UniformTest1)
    with pytest.warns():
        assert issubclass(plist.find_distribution("core.uniform", var_type="continuous", version="1.2"), UniformTest11)
    with pytest.raises(ValueError):
        plist.find_distribution("core.uniform", var_type="continuous", version="0.9")
    with pytest.warns():
        assert issubclass(plist.find_distribution("core.uniform", var_type="continuous", version="2.1"), UniformTest2)

    plist = DistributionProviderList(LegacyOnly)
    with pytest.warns():
        assert issubclass(plist.find_distribution("core.uniform", var_type="continuous"), UniformTest2)
