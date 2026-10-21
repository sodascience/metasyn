from pathlib import Path

import polars as pl
import pytest
from pytest import mark
from datetime import date, datetime, time

from metasyn.builder import (
    DistributionRecipe,
    FindDistributionRecipe,
    FitLog,
    FitterRecipe,
    MetaFrameBuilder,
    MultiFrameBuilder,
    UnqFindDistributionRecipe,
    VarBuilder,
)
from metasyn.demo import demo_data
from metasyn.distribution import (
    ContinuousConstantDistribution,
    DiscreteConstantDistribution,
    DiscreteConstantFitter,
    IfThenElse,
    RegexDistribution,
    RegexFitter,
)
from metasyn.distribution.base import BaseFitter, VarLog
from metasyn.metaframe import MetaFrame
from metasyn.privacy import BasicPrivacy
from metasyn.registry import DistributionRegistry


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

def test_varbuilder_error():
    vb = VarBuilder(series=ContinuousConstantDistribution(1.0))
    with pytest.raises(ValueError):
        vb.series
    builder = MetaFrameBuilder()
    vb.mf_builder = builder
    with pytest.raises(ValueError):
        vb.series
    builder.n_rows = 10
    vb.distribution = 10
    with pytest.raises(TypeError):
        vb.recipe

def test_builder_without_progress_bar():
    builder = MetaFrameBuilder(n_rows=10)
    builder.add_column("test")
    builder["test"].distribution = ContinuousConstantDistribution(1.0)
    assert isinstance(builder.fit(progress_bar=False), MetaFrame)

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
    print(builder["Int64"]._distribution, builder["Int64"]._dtype, builder["Int64"].mf_builder,
          builder["Int64"].series, builder["Int64"].dtype)
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

def test_class_distribution():
    series = pl.Series([1, 1, 1])
    vb = VarBuilder(series, "test", distribution=DiscreteConstantDistribution)
    mvar = vb.fit()
    assert isinstance(mvar.distribution, DiscreteConstantDistribution)
    assert mvar.distribution.value == 1

def test_find_fitter_error(builder):
    bld = MetaFrameBuilder()
    bld.add_column("Int64")
    bld.var_builders["Int64"] = builder["Int64"]
    bld["Int64"].distribution = "some_interesting_name"
    bld["Int64"].series = builder["Int64"].series
    with pytest.raises(ValueError):
        bld.fit()
    bld["Int64"].distribution = "regex"
    with pytest.raises(ValueError):
        bld.fit()

def test_fit_log_integration():
    builder = MetaFrameBuilder()
    builder.add_dataframe(demo_data("titanic")[:50])
    builder["Fare"].distribution = ContinuousConstantDistribution(10)
    builder["PassengerId"].distribution = {"unique": True}
    builder["Age"].fitter = DiscreteConstantFitter(BasicPrivacy())
    builder["Cabin"].fitter = RegexFitter(BasicPrivacy(), count_thres=20, method="fast")
    builder.fit()
    assert isinstance(builder.fit_log, FitLog)
    assert isinstance(builder.fit_log["Age"], VarLog)
    assert builder.fit_log["Fare"].created_by[0] == "user"
    assert builder.fit_log["Fare"].recipe[0] == "User defined distribution."
    assert builder.fit_log["PassengerId"].created_by[0] == "metasyn"
    assert isinstance(builder.fit_log["PassengerId"].fitter[0], BaseFitter)
    assert len(builder.fit_log["PassengerId"].bic) == 1
    assert builder.fit_log["Cabin"].method[0].startswith("Fitting regex using method 'fast'") == 1
    assert builder.fit_log["Cabin"].privacy[0].startswith("The privacy was done using the basic")

def test_fit_log(tmpdir):
    fitlog = FitLog()
    fitlog.add_col("test")
    fitlog.add_col("test2")
    fitlog["test"].add(method="blabla")
    fitlog.save_csv(tmpdir / "test.csv")
    assert (tmpdir / "test.csv").isfile()
    fitlog.save_csv(tmpdir / "test.md")
    assert (tmpdir / "test.md").isfile()
    fitlog.reset()
    assert len(fitlog["test"].method) == 0
    assert isinstance(str(fitlog), str)
    assert isinstance(fitlog.to_dataframe(), pl.DataFrame)

def test_add_none_multiframe():
    builder = MultiFrameBuilder()
    with pytest.raises(ValueError):
        builder.add_dataframe(None, "blable", None)


class PrivacyA(BasicPrivacy):
    pass

class PrivacyA1(BasicPrivacy):
    pass

class PrivacyB1(BasicPrivacy):
    pass

class PrivacyAny(BasicPrivacy):
    pass

def test_privacy_distributions():
    multi_builder = MultiFrameBuilder()
    multi_builder.add_dataframe(None, "builder_A", n_rows=10)
    multi_builder.add_dataframe(None, "builder_B", n_rows=10)
    multi_builder["builder_A"].add_column("col_A1", dtype="continuous")
    multi_builder["builder_A"].add_column("col_A2", dtype="continuous")
    multi_builder["builder_B"].add_column("col_B1", dtype="continuous")
    multi_builder["builder_B"].add_column("col_B2", dtype="continuous")

    multi_builder["builder_A"].privacy = PrivacyA()
    multi_builder["builder_A"]["col_A1"].privacy = PrivacyA1()
    multi_builder["builder_B"]["col_B1"].privacy = PrivacyB1()
    multi_builder.privacy = PrivacyAny()

    assert isinstance(multi_builder["builder_A"]["col_A1"].privacy, PrivacyA1)
    assert isinstance(multi_builder["builder_A"]["col_A2"].privacy, PrivacyA)
    assert isinstance(multi_builder["builder_B"]["col_B1"].privacy, PrivacyB1)
    assert isinstance(multi_builder["builder_B"]["col_B2"].privacy, PrivacyAny)

    multi_builder["builder_A"].set_default_distribution("continuous",
                                                        ContinuousConstantDistribution(0.0))
    multi_builder.set_default_distribution("continuous",
                                           ContinuousConstantDistribution(1.0))

    assert multi_builder["builder_A"]["col_A1"].distribution.value == 0.0
    assert multi_builder["builder_B"]["col_B1"].distribution.value == 1.0


@mark.parametrize(
    "list_a,list_b,dtype",
    [
        ([1, 2, 2], ["a", "b", "c"], "Object"),
        ([1, 2, 3], [0.3, 0.5, 0.6], "Object"),
        (["a", "b", "c"], 3*[date(1993, 4, 12)], "Object"),
        (["a", "b", "c"], 3*[time(14, 5)], "Object"),
        (["a", "b", "c"], 3*[datetime(1993, 4, 7, 10, 26)], "Object"),
        (["a", "b", "c"], [None, None, None], "String"),
        ([1, 2, 3], [None, None, None], "Int64"),
    ])
def test_mixed_distributions(list_a, list_b, dtype):
    pl_dtype = getattr(pl, dtype)
    series = pl.Series(list_a + list_b, dtype=pl_dtype)
    builder = VarBuilder(series, "test")
    mvar = builder.fit()
    assert mvar.dtype == dtype
    if dtype == "Object":
        assert isinstance(mvar.distribution, IfThenElse)
    else:
        assert not isinstance(mvar.distribution, IfThenElse)
    assert mvar.draw_series(10).dtype == pl_dtype

def test_unique_mixed_distributions():
    series = pl.Series([1, 2, "a", "b"], dtype=pl.Object)
    builder = VarBuilder(series, "test", distribution={"unique": True})
    mvar = builder.fit()
    assert isinstance(mvar.distribution, IfThenElse)
    assert mvar.dtype == "Object"

@mark.parametrize("save_meth", ["save_csv", "save_md"])
def test_write_fitlog(save_meth, tmpdir):
    builder = MetaFrameBuilder(auto_add_column=True, n_rows=10)
    builder["a"].distribution = ContinuousConstantDistribution(10.3)
    builder.fit()
    getattr(builder.fit_log, save_meth)(tmpdir / "test")
    assert (tmpdir/"test").isfile()
    assert any(len(getattr(builder.fit_log["a"], attr))!=0 for attr in builder.fit_log["a"].attrs)
    assert len(str(builder.fit_log["a"])) > 0
    builder.fit_log.reset()
    assert all(len(getattr(builder.fit_log["a"], attr))==0 for attr in builder.fit_log["a"].attrs)

def test_dtype_none():
    builder = VarBuilder()
    assert builder.dtype is None
