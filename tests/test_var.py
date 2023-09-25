import json

import pandas as pd
import polars as pl
import numpy as np
from metasyn.var import MetaVar
from metasyn.distribution import NormalDistribution, RegexDistribution, UniqueRegexDistribution
from metasyn.distribution import DiscreteUniformDistribution
from metasyn.distribution import UniformDistribution
from pytest import mark, raises
from metasyn.metaframe import _jsonify
from metasyn.distribution.discrete import UniqueKeyDistribution
from metasyn.distribution.continuous import TruncatedNormalDistribution
from metasyn.distribution.categorical import MultinoulliDistribution


def _series_drop_nans(series):
    if isinstance(series, pl.Series):
        return series.drop_nulls()
    return series.dropna()


def _series_element_classname(series):
    series = _series_drop_nans(series)
    if isinstance(series, pl.Series):
        return series[0].__class__.__name__
    return series.iloc[0].__class__.__name__


def check_var(series, var_type, temp_path):
    def check_similar(series_a, series_b):
        assert isinstance(series_a, (pd.Series, pl.Series))
        assert isinstance(series_b, (pd.Series, pl.Series))
        assert len(series_a) == len(series_b)
        base_type_a = _series_element_classname(series_a)
        base_type_b = _series_element_classname(series_b)
        if type(series_a) == type(series_b):
            assert base_type_a == base_type_b
        assert (len(series_a)-len(_series_drop_nans(series_a)) > 0) == (len(series_b) - len(_series_drop_nans(series_b)) > 0)

    assert isinstance(series, (pd.Series, pl.Series))
    var = MetaVar.detect(series)
    assert isinstance(str(var), str)
    assert "Proportion of Missing Values" in str(var)

    with raises(ValueError):
        var.draw_series(100)
    var.fit()
    new_series = var.draw_series(len(series))
    check_similar(series, new_series)
    assert var.var_type == var_type
    assert var_type in var.distribution.var_type

    new_var = MetaVar.from_dict(var.to_dict())
    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict.update({"type": "unknown"})
        MetaVar.from_dict(var_dict)

    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict["distribution"].update({"implements": "unknown"})
        MetaVar.from_dict(var_dict)

    newer_series = new_var.draw_series(len(series))
    check_similar(newer_series, series)
    with raises(ValueError):
        new_var.fit()

    assert type(new_var) == type(var)
    assert new_var.dtype == var.dtype
    assert var_type == new_var.var_type

    # Write to JSON file and read it back
    tmp_fp = temp_path / "tmp.json"
    with open(tmp_fp, "w") as f:
        json.dump(_jsonify(var.to_dict()), f)

    with open(tmp_fp, "r") as f:
        new_var = MetaVar.from_dict(json.load(f))
    check_similar(series, new_var.draw_series(len(series)))

    assert type(new_var) == type(var)
    assert new_var.dtype == var.dtype
    assert new_var.var_type == var_type

    return new_series


@mark.parametrize(
    "series",
    [
        pd.Series(np.random.choice(["a", "b", "c", None], size=100), dtype="category"),
        pl.Series(np.random.choice(["a", "b", "c", None], size=100).tolist(), dtype=pl.Categorical)
    ]
)
def test_categorical(tmp_path, series):
    # series = pd.Series(np.random.choice(["a", "b", "c", None], size=100), dtype="category")
    new_series = check_var(series, "categorical", tmp_path)
    assert set(_series_drop_nans(series)) == set(np.unique(_series_drop_nans(new_series)))


@mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "int"])
def test_integer(dtype, tmp_path):
    series = pd.Series([np.random.randint(0, 10) for _ in range(100)], dtype=dtype)
    new_series = check_var(series, "discrete", tmp_path)
    assert new_series.min() >= 0
    assert new_series.max() < 10


@mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64"])
def test_nullable_integer(dtype, tmp_path):
    series = pd.Series([np.random.randint(0, 10) if np.random.rand() > 0.5 else None
                        for _ in range(100)], dtype=dtype)
    new_series = check_var(series, "discrete", tmp_path)
    assert new_series.min() >= 0
    assert new_series.max() < 10


@mark.parametrize(
    "series_type",
    [pl.Series, pd.Series],
)
def test_float(tmp_path, series_type):
    np.random.seed(3727442)
    series = series_type([np.random.rand() for _ in range(10000)])
    new_series = check_var(series, "continuous", tmp_path)
    assert new_series.min() > 0
    assert new_series.max() < 1

    series = pd.Series(np.random.randn(1000))
    check_var(series, "continuous", tmp_path)


@mark.parametrize(
    "series_type",
    [pl.Series, pd.Series],
)
def test_string(tmp_path, series_type):
    series = series_type(np.random.choice(["a", "b", "c", None], size=100).tolist())
    new_series = check_var(series, "string", tmp_path)
    assert set(np.unique(_series_drop_nans(series))) == set(np.unique(_series_drop_nans(new_series)))


@mark.parametrize(
    "series_type",
    [pl.Series, pd.Series]
)
def test_bool(tmp_path, series_type):
    series = series_type(np.random.choice([True, False], size=100))
    with raises(ValueError):
        check_var(series, "categorical", tmp_path)


@mark.parametrize(
    "prop_missing",
    [-1, -0.1, 1.2],
)
def test_invalid_prop(prop_missing):
    with raises(ValueError):
        MetaVar("continuous")
    with raises(ValueError):
        MetaVar("continuous", prop_missing=prop_missing)


@mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({
            "int": pd.Series([np.random.randint(0, 10) for _ in range(100)]),
            "float": pd.Series([np.random.rand() for _ in range(100)])
        }),
        pl.DataFrame({
            "int": [np.random.randint(0, 10) for _ in range(100)],
            "float": [np.random.rand() for _ in range(100)]
        })
    ]
)
def test_dataframe(dataframe):
    variables = MetaVar.detect(dataframe)
    assert len(variables) == 2
    assert isinstance(variables, list)
    assert variables[0].var_type == "discrete"
    assert variables[1].var_type == "continuous"


@mark.parametrize(
    "series",
    [pd.Series([np.random.rand() for _ in range(5000)]),
     pl.Series([np.random.rand() for _ in range(5000)])]
)
def test_manual_fit(series):
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, (UniformDistribution, TruncatedNormalDistribution))
    var.fit("normal")
    assert isinstance(var.distribution, NormalDistribution)
    var.fit(UniformDistribution)
    assert isinstance(var.distribution, UniformDistribution)
    var.fit(NormalDistribution(0, 1))
    assert isinstance(var.distribution, NormalDistribution)
    with raises(TypeError):
        var.fit(10)


@mark.parametrize(
    "series",
    [pd.Series([pd.NA for _ in range(10)]),
     pl.Series([None for _ in range(10)])]
)
def test_na_zero(series):
    var = MetaVar.detect(series)
    var.fit()
    assert var.var_type == "continuous"
    assert var.prop_missing == 1.0


@mark.parametrize(
    "series",
    [pd.Series(np.array([pd.NA if i != 0 else 1.0 for i in range(10)])),
     pl.Series([None if i != 0 else 1.0 for i in range(10)])]
)
def test_na_one(series):
    var = MetaVar.detect(series)
    var.fit()
    assert var.var_type == "continuous"
    assert abs(var.prop_missing-0.9) < 1e7


@mark.parametrize(
    "series",
    [pd.Series(np.array([np.nan if i < 2 else 0.123*i for i in range(10)])),
     pl.Series([None if i < 2 else 0.123*i for i in range(10)])]
)
def test_na_two(series):
    var = MetaVar.detect(series)
    var.fit()
    assert var.var_type == "continuous"
    assert abs(var.prop_missing-0.8) < 1e7


@mark.parametrize(
    "series",
    [pd.Series(np.random.randint(0, 100000, size=1000)),
     pl.Series(np.random.randint(0, 100000, size=1000))]
)
def test_manual_unique_integer(series):
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, DiscreteUniformDistribution)
    var.fit(unique=True)
    assert isinstance(var.distribution, UniqueKeyDistribution)


@mark.parametrize(
    "series",
    [
        pd.Series(["x213", "2dh2", "4k2kk"]),
        pl.Series(["x213", "2dh2", "4k2kk"]),
    ]
)
def test_manual_unique_string(series):
    # series = pd.Series(["x213", "2dh2", "4k2kk"])
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, RegexDistribution)
    var.fit(unique=True)
    assert isinstance(var.distribution, UniqueRegexDistribution)


@mark.parametrize(
    "series",
    [
        pl.Series([2]*100 + [4]*100 + [12394]*100),
        pl.Series([-123]*12 + [123]*12),
        # pl.Series(["2"]*100 + ["5"]*100 + ["123"]*100),
        pl.Series(["2"]*100 + ["5"]*100 + ["123"]*100, dtype=pl.Categorical)
    ]
)
def test_int_multinoulli(series):
    var = MetaVar.detect(series)
    var.fit()
    assert isinstance(var.distribution, MultinoulliDistribution)
