"""Module to test the MetaVar object.

Most of the test functions try to infer the correct distribution from a series.
"""
import datetime as dt
import json
import string

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pytest import mark, raises

from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.normal import NormalDistribution, TruncatedNormalDistribution
from metasyn.distribution.regex import RegexDistribution, UniqueRegexDistribution
from metasyn.distribution.uniform import DiscreteUniformDistribution, ContinuousUniformDistribution
from metasyn.distribution.uniquekey import UniqueKeyDistribution
from metasyn.metaframe import _jsonify
from metasyn.var import MetaVar


def _series_drop_nans(series):
    if isinstance(series, pl.Series):
        return series.drop_nulls()
    return series.dropna()


def _series_element_classname(series, all_nan):
    if not all_nan:
        series = _series_drop_nans(series)
    if isinstance(series, pl.Series):
        return series[0].__class__.__name__
    return series.iloc[0].__class__.__name__


def check_var(series, var_type, temp_path, all_nan=False):
    """Checker for all variables whether series are inferred correctly."""
    def check_similar(series_a, series_b):
        assert isinstance(series_a, (pd.Series, pl.Series))
        assert isinstance(series_b, (pd.Series, pl.Series))
        assert len(series_a) == len(series_b)
        base_type_a = _series_element_classname(series_a, all_nan)
        base_type_b = _series_element_classname(series_b, all_nan)
        if type(series_a) is type(series_b):
            assert base_type_a == base_type_b
        has_nans_a = len(series_a) - len(_series_drop_nans(series_a)) > 0
        has_nans_b = len(series_b) - len(_series_drop_nans(series_b)) > 0
        assert has_nans_a == has_nans_b

    def check_random_draw(var, n_series):
        series_1 = var.draw_series(n_series, 1234)
        series_2 = var.draw_series(n_series, 1234)
        assert all(_series_drop_nans(series_1) == _series_drop_nans(series_2))

    assert isinstance(series, (pd.Series, pl.Series))

    var = MetaVar.fit(series)
    new_series = var.draw_series(len(series), 5123)
    check_random_draw(var, len(series))
    print(new_series)
    check_similar(series, new_series)
    assert var.var_type == var_type
    assert var_type in var.distribution.var_type
    assert var.creation_method["created_by"] == "metasyn"


    new_var = MetaVar.from_dict(var.to_dict())
    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict.update({"type": "unknown"})
        print(MetaVar.from_dict(var_dict))

    with raises(ValueError):
        var_dict = var.to_dict()
        var_dict["distribution"].update({"implements": "unknown"})
        MetaVar.from_dict(var_dict)

    newer_series = new_var.draw_series(len(series), 6789)
    check_similar(newer_series, series)

    assert type(new_var) is type(var)
    assert new_var.dtype == var.dtype
    assert var_type == new_var.var_type

    # Write to JSON file and read it back
    tmp_fp = temp_path / "tmp.json"
    with open(tmp_fp, "w") as f:
        json.dump(_jsonify(var.to_dict()), f)

    with open(tmp_fp, "r") as f:
        new_var = MetaVar.from_dict(json.load(f))
    check_similar(series, new_var.draw_series(len(series), 8234))

    assert type(new_var) is type(var)
    assert new_var.dtype == var.dtype
    assert new_var.var_type == var_type
    assert new_var.creation_method["created_by"] == "metasyn"

    return new_series


@mark.parametrize(
    "series",
    [
        pd.Series(np.random.choice(["a", "b", "c", None], size=100), dtype="category"),
        pl.Series(np.random.choice(["a", "b", "c", None], size=100).tolist(), dtype=pl.Categorical)
    ]
)
def test_categorical(tmp_path, series):
    """Test the inference of categorical variables."""
    new_series = check_var(series, "categorical", tmp_path)
    assert set(_series_drop_nans(series)) == set(np.unique(_series_drop_nans(new_series)))

@mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "int"])
def test_integer(dtype, tmp_path):
    """Test the inference of integer variables."""
    series = pd.Series([np.random.randint(0, 10) for _ in range(300)], dtype=dtype)
    new_series = check_var(series, "discrete", tmp_path)
    assert new_series.min() >= 0
    assert new_series.max() < 10


@mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64"])
def test_nullable_integer(dtype, tmp_path):
    """Test for pandas nullable integer data types (no applicable to polars)."""
    series = pd.Series([np.random.randint(0, 10) if np.random.rand() > 0.5 else None
                        for _ in range(500)], dtype=dtype)
    new_series = check_var(series, "discrete", tmp_path)
    assert new_series.min() >= 0
    assert new_series.max() < 10

@mark.parametrize(
    "dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int32, pl.Int64,
              pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
)
def test_polars_discrete(dtype, tmp_path):
    """Test for all different variables of polars discrete types."""
    series = pl.Series("ser", [np.random.randint(0, 10) for _ in range(100)], dtype=dtype)
    new_series = check_var(series, "discrete", tmp_path)
    assert new_series.min() >= 0
    assert new_series.max() < 10

@mark.parametrize(
    "dtype", [pl.Float32, pl.Float64]
)
def test_polars_continuous(dtype, tmp_path):
    """Test for all polars continuous data types."""
    series = pl.Series("ser", np.random.rand(100), dtype=dtype)
    new_series = check_var(series, "continuous", tmp_path)
    assert new_series.min() > -0.5
    assert new_series.max() < 1.5


@mark.parametrize(
    "dtype", [pl.String, pl.Utf8]
)
def test_polars_string(dtype, tmp_path):
    """Test polars string types."""
    series = pl.Series("series", string.printable, dtype=dtype)
    check_var(series, "string", tmp_path)

@mark.parametrize(
    "dtype", [pl.Categorical]
)
def test_polars_categorical(dtype, tmp_path):
    """Test whether polars categorical data types are parsed correctly."""
    series = pl.Series("series", string.printable, dtype=dtype)
    check_var(series, "categorical", tmp_path)

@mark.parametrize(
    "series_type",
    [pl.Series, pd.Series],
)
def test_float(tmp_path, series_type):
    """Test for continuous variables, pandas or polars."""
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
    """Test polars/pandas string type fitting."""
    series = series_type(np.random.choice(["a", "b", "c", None], size=100).tolist())
    new_series = check_var(series, "string", tmp_path)
    unq_values = set(np.unique(_series_drop_nans(series)))
    new_unq_values = set(np.unique(_series_drop_nans(new_series)))
    assert unq_values == new_unq_values

@mark.parametrize("dtype,var_type", [
    (pl.Float32, "continuous"),
    (pl.Float64, "continuous"),
    (pl.Int32, "discrete"),
    (pl.Int64, "discrete"),
    (pl.Categorical, "categorical"),
    (pl.Utf8, "string")])
def test_na(tmp_path, dtype, var_type):
    """Test the various NA type distributions."""
    series = pl.Series("x", [None, None, None], dtype=dtype)
    print(series)
    new_series = check_var(series, var_type, tmp_path, all_nan=True)
    assert all(x is None for x in new_series)


@mark.parametrize(
    "series_type",
    [pl.Series, pd.Series]
)
def test_bool(tmp_path, series_type):
    """Test whether booleans are correctly handled as categorical variables."""
    series = series_type(np.random.choice([True, False], size=100))
    check_var(series, "categorical", tmp_path)
    var = MetaVar.fit(series)
    new_series = var.draw_series(10, 1234)
    assert new_series.dtype == pl.Boolean


@mark.parametrize(
    "prop_missing",
    [-1, -0.1, 1.2],
)
def test_invalid_prop(prop_missing):
    """Ensure that errors are thrown for prop_missing outside of [0, 1]."""
    MetaVar("test", "discrete", DiscreteUniformDistribution.default_distribution())
    with raises(ValueError):
        MetaVar("test", "discrete", DiscreteUniformDistribution.default_distribution(),
                prop_missing=prop_missing)

@mark.parametrize(
    "series,var_type",
    [
        (pl.Series([1, 2, 3]), "discrete"),
        (pd.Series([1, 2, 3]), "discrete"),
        (pl.Series([1.0, 2.0, 3.0]), "continuous"),
        (pl.Series(["1", "2", "3"]), "string"),
        (pl.Series(["1", "2", "3"], dtype=pl.Categorical), "categorical"),
        (pl.Series([dt.time.fromisoformat("10:38:12"), dt.time.fromisoformat("12:52:11")]),
            "time"),
        (pl.Series([dt.datetime.fromisoformat("2022-07-15T10:39:36"),
                    dt.datetime.fromisoformat("2022-08-15T10:39:36")]),
            "datetime"),
        (pl.Series([dt.date.fromisoformat("1903-07-15"), dt.date.fromisoformat("1940-07-16")]),
            "date"),
    ]
)
def test_get_var_type(series, var_type):
    """Test get_var_type method of MetaVar."""
    assert MetaVar.get_var_type(series) == var_type


def test_unsupported_type():
    series = pl.Series([MetaVar])
    with pytest.raises(TypeError):
        MetaVar.get_var_type(series)


@mark.parametrize(
    "series",
    [pd.Series([np.random.rand() for _ in range(5000)]),
     pl.Series([np.random.rand() for _ in range(5000)])]
)
def test_manual_fit(series):
    """Test adding dist_spec to MetaVar.fit call."""
    var = MetaVar.fit(series)
    assert isinstance(var.distribution, (ContinuousUniformDistribution, TruncatedNormalDistribution))
    var = MetaVar.fit(series, dist_spec={"implements": "normal"})
    assert isinstance(var.distribution, NormalDistribution)
    var = MetaVar.fit(series, dist_spec=ContinuousUniformDistribution)
    assert isinstance(var.distribution, ContinuousUniformDistribution)
    var = MetaVar.fit(series, dist_spec=NormalDistribution(0, 1))
    assert isinstance(var.distribution, NormalDistribution)
    with raises(TypeError):
        var.fit(10)


@mark.parametrize(
    "series",
    [pd.Series([pd.NA for _ in range(10)]),
     pl.Series([None for _ in range(10)])]
)
def test_na_zero(series):
    """Test whether fitting a series with only NA works."""
    var = MetaVar.fit(series)
    assert var.var_type == "continuous"
    assert var.prop_missing == 1.0


@mark.parametrize(
    "series",
    [pd.Series(np.array([pd.NA if i != 0 else 1.0 for i in range(10)])),
     pl.Series([None if i != 0 else 1.0 for i in range(10)])]
)
def test_na_one(series):
    """Test whether fitting a series with exactly one non-NA values works."""
    var = MetaVar.fit(series)
    assert var.var_type == "continuous"
    assert abs(var.prop_missing-0.9) < 1e7


@mark.parametrize(
    "series",
    [pd.Series(np.array([np.nan if i < 2 else 0.123*i for i in range(10)])),
     pl.Series([None if i < 2 else 0.123*i for i in range(10)])]
)
def test_na_two(series):
    """Test whether fitting a series with exactly two non-NA values works."""
    var = MetaVar.fit(series)
    assert var.var_type == "continuous"
    assert abs(var.prop_missing-0.8) < 1e7


@mark.parametrize(
    "series",
    [pd.Series(np.random.randint(0, 1000, size=500)),
     pl.Series(np.random.randint(0, 1000, size=500))]
)
def test_manual_unique_integer(series):
    """Test whether discrete uniform/unique key are correctly inferred."""
    var = MetaVar.fit(series)
    assert isinstance(var.distribution, DiscreteUniformDistribution)
    var = MetaVar.fit(series, dist_spec = {"unique": True})
    assert isinstance(var.distribution, UniqueKeyDistribution)


@mark.parametrize(
    "series",
    [
        pd.Series(["x213", "2dh2", "4k2kk"]),
        pl.Series(["x213", "2dh2", "4k2kk"]),
    ]
)
def test_manual_unique_string(series):
    """Test which regexmodel is used dependent on the uniqueness."""
    var = MetaVar.fit(series)
    assert isinstance(var.distribution, RegexDistribution)
    var = MetaVar.fit(series, dist_spec={"unique": True})
    assert isinstance(var.distribution, UniqueRegexDistribution)


@mark.parametrize(
    "series",
    [
        pl.Series([2]*100 + [4]*100 + [12394]*100),
        pl.Series([-123]*12 + [123]*12),
        pl.Series(["2"]*100 + ["5"]*100 + ["123"]*100, dtype=pl.Categorical)
    ]
)
def test_int_multinoulli(series):
    """Test whether the multinoulli distribution is correctly inferred."""
    var = MetaVar.fit(series)
    assert isinstance(var.distribution, MultinoulliDistribution)


def test_error_multinoulli():
    """Test whether incorrect probabilities give a warning/error."""
    with raises(ValueError):
        MultinoulliDistribution(["1", "2"], [-0.1, 1.1])
    with pytest.warns():
        MultinoulliDistribution(["1", "2"], [0.1, 0.2])
