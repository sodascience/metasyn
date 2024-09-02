import pytest

from metasyn.distribution import MultinoulliDistribution
from metasyn.distribution.base import metadist
from metasyn.privacy import BasicPrivacy, get_privacy


@metadist(privacy="test")
class OtherMultinoulli(MultinoulliDistribution):
    pass


def test_base_privacy():
    privacy = BasicPrivacy()
    priv_dict = privacy.to_dict()
    assert isinstance(priv_dict, dict)
    assert priv_dict["name"] == "none"
    assert privacy.name == "none"
    assert isinstance(priv_dict["parameters"], dict)
    assert len(priv_dict["parameters"]) == 0
    assert privacy.is_compatible(MultinoulliDistribution)
    assert privacy.is_compatible(MultinoulliDistribution(["1"], [1.0]))
    assert not privacy.is_compatible(OtherMultinoulli)


def test_import_error():
    with pytest.raises(ImportError):
        get_privacy("nonexistentprivacyplugin")
