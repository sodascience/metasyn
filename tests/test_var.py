from pathlib import Path
import json

import pandas as pd
import numpy as np
from metasynth.var import MetaVar
from metasynth.distribution import MultinoulliDistribution, NormalDistribution,\
    RegexDistribution
from metasynth.distribution import DiscreteUniformDistribution
from metasynth.distribution import UniformDistribution
from pytest import mark, raises
from metasynth.dataset import _jsonify
from metasynth.distribution.discrete import UniqueKeyDistribution
from metasynth.distribution.regex import UniqueRegexDistribution
from metasynth.distribution.base import CategoricalDistribution,\
    DiscreteDistribution, ContinuousDistribution


def check_var(series, var_type):
    def check_similar(series_a, series_b):
        assert isinstance(series_a, pd.Series)
        assert isinstance(series_b, pd.Series)
        assert len(series_a) == len(series_b)
        base_type_a = series_a.dropna().iloc[0].__class__.__name__
        base_type_b = series_b.dropna().iloc[0].__class__.__name__
        assert base_type_a == base_type_b
        assert (len(series_a)-len(series_a.dropna()) > 0) == (len(series_b) - len(series_b.dropna()) > 0)

    assert isinstance(series, pd.Series)
    var = MetaVar.detect(series)
    assert isinstance(str(var), str)
    assert "prop_missing" in str(var)

    with raises(ValueError):
        var.draw_series(100)
    var.fit()
    new_series = var.draw_series(len(series))
    check_similar(series, new_series)
    assert var.var_type == var_type
    assert var.distribution.var_type == var_type
    # assert isinstance(var.distribution, dist_class)

    new_var = MetaVar.from_dict(var.to_dict())
    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict.update({"type": "unknown"})
        MetaVar.from_dict(var_dict)

    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict["distribution"].update({"name": "unknown"})
        MetaVar.from_dict(var_dict)

    newer_series = new_var.draw_series(len(series))
    check_similar(newer_series, series)
    with raises(ValueError):
        new_var.fit()

    assert type(new_var) == type(var)
    assert new_var.dtype == var.dtype
    assert new_var.var_type == var_type

    # Write to JSON file and read it back
    tmp_fp = Path("tests", "data", "tmp.json")
    with open(tmp_fp, "w") as f:
        json.dump(_jsonify(var.to_dict()), f)

    with open(tmp_fp, "r") as f:
        new_var = MetaVar.from_dict(json.load(f))
    check_similar(series, new_var.draw_series(len(series)))

    assert type(new_var) == type(var)
    assert new_var.dtype == var.dtype
    assert new_var.var_type == var_type

    return new_series


def test_categorical():
    series = pd.Series(np.random.choice(["a", "b", "c", None], size=100), dtype="category")
    new_series = check_var(series, "categorical")
    assert set(np.unique(series.dropna())) == set(np.unique(new_series.dropna()))


@mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "int"])
def test_integer(dtype):
    series = pd.Series([np.random.randint(0, 10) for _ in range(100)], dtype=dtype)
    new_series = check_var(series, "discrete")
    assert np.min(new_series) >= 0
    assert np.max(new_series) < 10


@mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64"])
def test_nullable_integer(dtype):
    series = pd.Series([np.random.randint(0, 10) if np.random.rand() > 0.5 else None
                        for _ in range(100)], dtype=dtype)
    new_series = check_var(series, "discrete")
    assert np.min(new_series) >= 0
    assert np.max(new_series) < 10


def test_float():
    np.random.seed(3727442)
    series = pd.Series([np.random.rand() for _ in range(10000)])
    new_series = check_var(series, "continuous")
    assert np.min(new_series) > 0
    assert np.max(new_series) < 1

    series = pd.Series(np.random.randn(1000))
    check_var(series, "continuous")


def test_string():
    series = pd.Series(np.random.choice(["a", "b", "c", None], size=100))
    new_series = check_var(series, "string")
    assert set(np.unique(series.dropna())) == set(np.unique(new_series.dropna()))


def test_bool():
    series = pd.Series(np.random.choice([True, False], size=100))
    with raises(ValueError):
        check_var(series, "categorical")


def test_dataframe():
    df = pd.DataFrame({
        "int": pd.Series([np.random.randint(0, 10) for _ in range(100)]),
        "float": pd.Series([np.random.rand() for _ in range(100)])
    })

    variables = MetaVar.detect(df)
    assert len(variables) == 2
    assert isinstance(variables, list)
    assert variables[0].var_type == "discrete"
    assert variables[1].var_type == "continuous"


def test_manual_fit():
    series = pd.Series([np.random.rand() for _ in range(500)])
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, UniformDistribution)
    var.fit("normal")
    assert isinstance(var.distribution, NormalDistribution)
    var.fit(UniformDistribution)
    assert isinstance(var.distribution, UniformDistribution)
    var.fit(NormalDistribution(0, 1))
    assert isinstance(var.distribution, NormalDistribution)
    with raises(TypeError):
        var.fit(10)


def test_manual_unique():
    series = pd.Series(np.random.randint(0, 100000, size=10))
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, DiscreteUniformDistribution)
    var.fit(unique=True)
    assert isinstance(var.distribution, UniqueKeyDistribution)

    series = pd.Series(["x213", "2dh2", "4k2kk"])
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, RegexDistribution)
    var.fit(unique=True)
    assert isinstance(var.distribution, UniqueRegexDistribution)
