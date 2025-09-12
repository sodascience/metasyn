"""Test whether the privacy functionality works."""
import pytest

from metasyn.distribution.base import metafit
from metasyn.distribution.categorical import MultinoulliFitter
from metasyn.privacy import BasicPrivacy, get_privacy


@metafit(privacy_type="test")
class OtherMultinoulli(MultinoulliFitter):
    """Test privacy version of multinoulli."""


def test_base_privacy():
    """Test whether the builtin version of privacy (no extra) works as expected."""
    privacy = BasicPrivacy()
    priv_dict = privacy.to_dict()
    assert isinstance(priv_dict, dict)
    assert priv_dict["name"] == "none"
    assert privacy.name == "none"
    assert isinstance(priv_dict["parameters"], dict)
    assert len(priv_dict["parameters"]) == 0
    assert privacy.is_compatible(MultinoulliFitter)
    assert not privacy.is_compatible(OtherMultinoulli)


def test_import_error():
    """Test whether trying to use a non-existing privacy type raises an error."""
    with pytest.raises(ImportError):
        get_privacy("nonexistentprivacyplugin")
