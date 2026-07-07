from pathlib import Path

import polars as pl
import pytest

from metasyn.builder import MetaFrameBuilder, VarBuilder
from metasyn.demo import demo_data
from metasyn.distribution import DiscreteConstantDistribution
from metasyn.privacy import BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.builder import FitterRecipe, DistributionRecipe, FindDistributionRecipe, UnqFindDistributionRecipe


@pytest.fixture(scope="module")
def builder():
    builder = MetaFrameBuilder()
    builder.add_dataframe(demo_data("test"))
    return builder

class OtherPrivacy(BasicPrivacy):
    pass

def test_builder_registry(builder):
    builder["Int64"].registry = DistributionRegistry([])
    assert builder["Int64"].registry.fitters == []
    assert len(builder["Int32"].registry.fitters) > 0
    builder["Int64"].registry = "builtin"
    assert isinstance(builder["Int64"].registry, DistributionRegistry)

def test_builder_series(builder):
    assert isinstance(builder["Int64"].series, pl.Series)
    builder["Int64"].series = DiscreteConstantDistribution(10)
    assert isinstance(builder["Int64"].series, pl.Series)

def test_builder_privacy(builder):
    builder.privacy = OtherPrivacy()
    assert isinstance(builder["Int64"].privacy, OtherPrivacy)
    builder["Int64"].privacy = BasicPrivacy()
    assert builder["Int64"].privacy.__class__ == BasicPrivacy

def test_add_column():
    builder = MetaFrameBuilder()
    builder.add_column("extra")
    assert isinstance(builder["extra"], VarBuilder)
    assert builder.columns == ["extra"]

def test_add_config():
    builder = MetaFrameBuilder()
    builder.add_config(Path("tests", "data", "no_data_config.toml"))
    assert "Cabin" in builder.columns
    assert isinstance(builder["Cabin"], VarBuilder)

    with pytest.raises(FileNotFoundError):
        builder.add_config("this_file_does_not_exist.toml")
    with pytest.raises(ValueError):
        builder.add_config(Path("tests", "data", "actually_a_csv_file.sav"))
    with pytest.raises(ValueError):
        builder.add_config(Path("tests", "data", "unsupported_config.toml"))

def test_default_distribution(builder):
    builder.defaults = {"distribution": {"discrete": DiscreteConstantDistribution(10)}}
    assert isinstance(builder["Int64"].distribution, DiscreteConstantDistribution)
    assert builder.get_default_distribution("continuous") is None
    builder.defaults = {}

def test_recipes(builder):
    builder["Int64"].distribution = DiscreteConstantDistribution
    assert isinstance(builder["Int64"].recipe, FitterRecipe)
    builder["Int64"].distribution = DiscreteConstantDistribution(10)
    assert isinstance(builder["Int64"].recipe, DistributionRecipe)
    builder["Int64"].distribution = {"unique": True}
    assert isinstance(builder["Int64"].recipe, FindDistributionRecipe)
    builder["Int64"].distribution = None
    assert isinstance(builder["Int64"].recipe, UnqFindDistributionRecipe)
    builder["Int64"].series = DiscreteConstantDistribution(10)
    assert isinstance(builder["Int64"].recipe, UnqFindDistributionRecipe)

def test_find_fitter_error(builder):
    bld = MetaFrameBuilder()
    bld.add_column("Int64")
    bld["Int64"] = builder["Int64"]
    bld["Int64"].distribution = "some_interesting_name"
    bld["Int64"].series = builder.series
    with pytest.raises(ValueError):
        bld.fit()
    bld["Int64"].distribution = "regex"
    with pytest.raises(ValueError):
        bld.fit()
