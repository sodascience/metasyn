from pathlib import Path

import pytest
from pytest import mark

from metasyn.config import MetaConfig, VarSpecAccess
from metasyn.distribution import UniformDistribution
from metasyn.privacy import BasePrivacy
from metasyn.varspec import DistributionSpec, VarSpec


@mark.parametrize(
    "input,error",
    [
        ("uniform", False),
        (None, False),
        ({"implements": {"uniform": False}}, False),
        (UniformDistribution, False),
        (UniformDistribution(0, 2), False),
        (DistributionSpec(), False),
        ({"fit_kwargs": {"param": 3}}, True),
        ({"version": "2.0"}, True),
        ({"parameters": {"param": 2}}, True),
        (1, True),
    ]
)
def test_dist_spec(input, error):
    if error:
        with pytest.raises(Exception):
            DistributionSpec.parse(input)
    else:
        dist_spec = DistributionSpec.parse(input)
        assert isinstance(dist_spec, DistributionSpec)

def test_var_spec():
    var_spec = VarSpec("test", privacy={"name": "none", "parameters": {}})
    assert var_spec.name == "test"
    assert isinstance(var_spec.privacy, BasePrivacy)
    with pytest.raises(ValueError):
        var_spec = VarSpec("test", data_free=True)
    var_spec = VarSpec("test")
    assert isinstance(var_spec.dist_spec, DistributionSpec)
    var_spec = VarSpec.from_dict({"name": "test"})
    assert var_spec.name == "test"
    var_spec = VarSpec.from_dict({"name": "test", "distribution": "uniform"})
    assert var_spec.dist_spec.implements == "uniform"


def test_meta_config_datafree():
    meta_config = MetaConfig.from_toml(Path("tests", "data", "no_data_config.toml"))
    assert meta_config.n_rows == 100
    assert len(meta_config.var_specs) == 3
    assert isinstance(meta_config.var_specs[0], VarSpec)
    assert meta_config.var_specs[0].privacy is None
    assert isinstance(meta_config.var_specs[0].dist_spec, DistributionSpec)
    assert isinstance(meta_config.to_dict(), dict)
    var_spec = meta_config.get("PassengerId")
    assert isinstance(var_spec, VarSpecAccess)

    assert var_spec.data_free is True
    var_spec = meta_config.get("unknown")
    assert var_spec.name == "unknown"

    all_var_spec = list(meta_config.iter_var())
    assert len(all_var_spec) == 3
    assert isinstance(all_var_spec[0], VarSpecAccess)
    assert all_var_spec[0].meta_config == meta_config
    assert len(list(meta_config.iter_var(exclude=["PassengerId"]))) == 2


def test_meta_config():
    meta_config = MetaConfig.from_toml(Path("tests", "data", "example_config.toml"))
    assert len(meta_config.var_specs) == 5
    var_spec = meta_config.get("Cabin")
    assert var_spec.data_free is False
    assert var_spec.var_type is None
    assert var_spec.dist_spec.implements == "core.regex"
