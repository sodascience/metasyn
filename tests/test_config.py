from pathlib import Path

import pytest
from pytest import mark

from metasyn.config import MetaConfig, VarConfigAccess
from metasyn.distribution import UniformDistribution
from metasyn.privacy import BasePrivacy
from metasyn.util import DistributionSpec, VarConfig


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

def test_var_config():
    var_cfg = VarConfig("test", privacy={"name": "none", "parameters": {}})
    assert var_cfg.name == "test"
    assert isinstance(var_cfg.privacy, BasePrivacy)
    with pytest.raises(ValueError):
        var_cfg = VarConfig("test", data_free=True)
    var_cfg = VarConfig("test")
    assert isinstance(var_cfg.dist_spec, DistributionSpec)
    var_cfg = VarConfig.from_dict({"name": "test"})
    assert var_cfg.name == "test"
    var_cfg = VarConfig.from_dict({"name": "test", "distribution": "uniform"})
    assert var_cfg.dist_spec.implements == "uniform"


def test_meta_config_datafree():
    meta_config = MetaConfig.from_toml(Path("tests", "data", "no_data_config.toml"))
    assert meta_config.n_rows == 100
    assert len(meta_config.var_configs) == 3
    assert isinstance(meta_config.var_configs[0], VarConfig)
    assert meta_config.var_configs[0].privacy is None
    assert isinstance(meta_config.var_configs[0].dist_spec, DistributionSpec)
    assert isinstance(meta_config.to_dict(), dict)
    var_cfg = meta_config.get("PassengerId")
    assert isinstance(var_cfg, VarConfigAccess)
    print(var_cfg.var_config)
    assert var_cfg.data_free is True
    var_cfg = meta_config.get("unknown")
    assert var_cfg.name == "unknown"

    all_var_cfg = list(meta_config.iter_var())
    assert len(all_var_cfg) == 3
    assert isinstance(all_var_cfg[0], VarConfigAccess)
    assert all_var_cfg[0].meta_config == meta_config
    assert len(list(meta_config.iter_var(exclude=["PassengerId"]))) == 2


def test_meta_config():
    meta_config = MetaConfig.from_toml(Path("tests", "data", "example_config.toml"))
    assert len(meta_config.var_configs) == 5
    var_cfg = meta_config.get("Cabin")
    assert var_cfg.data_free is False
    assert var_cfg.var_type is None
    assert var_cfg.dist_spec.implements == "core.regex"
