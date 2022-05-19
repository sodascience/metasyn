import pandas as pd
import numpy as np
from metasynth.var import MetaVar, CategoricalVar, IntVar, FloatVar, StringVar
from metasynth.distribution import CatFreqDistribution, NormalDistribution,\
    StringFreqDistribution
from metasynth.distribution import DiscreteUniformDistribution
from metasynth.distribution import UniformDistribution
from pytest import mark


def check_var(series, var_type, dist_class):
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
    var.fit()
    new_series = var.draw_series(100)
    check_similar(series, new_series)
    assert isinstance(var, var_type)
    assert isinstance(var.distribution, dist_class)

    new_var = MetaVar.from_dict(var.to_dict())
    newer_series = new_var.draw_series(100)
    check_similar(newer_series, series)

    assert type(new_var) == type(var)
    assert new_var.dtype == var.dtype
    assert isinstance(new_var, var_type)
    return new_series


def test_categorical():
    series = pd.Series(np.random.choice(["a", "b", "c", None], size=100), dtype="category")
    new_series = check_var(series, CategoricalVar, CatFreqDistribution)
    assert set(np.unique(series.dropna())) == set(np.unique(new_series.dropna()))


@mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "int"])
def test_integer(dtype):
    series = pd.Series([np.random.randint(0, 10) for _ in range(100)], dtype=dtype)
    new_series = check_var(series, IntVar, DiscreteUniformDistribution)
    assert np.min(new_series) >= 0
    assert np.max(new_series) < 10


@mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64"]) # 
def test_nullable_integer(dtype):
    series = pd.Series([np.random.randint(0, 10) if np.random.rand() > 0.5 else None for _ in range(100)], dtype=dtype)
    new_series = check_var(series, IntVar, DiscreteUniformDistribution)
    assert np.min(new_series) >= 0
    assert np.max(new_series) < 10


def test_float():
    series = pd.Series([np.random.rand() for _ in range(100)])
    new_series = check_var(series, FloatVar, UniformDistribution)
    assert np.min(new_series) > 0
    assert np.max(new_series) < 1

    series = pd.Series(np.random.randn(100))
    check_var(series, FloatVar, NormalDistribution)


def test_string():
    series = pd.Series(np.random.choice(["a", "b", "c", None], size=100))
    new_series = check_var(series, StringVar, StringFreqDistribution)
    assert set(np.unique(series.dropna())) == set(np.unique(new_series.dropna()))


def test_dataframe():
    df = pd.DataFrame({
        "int": pd.Series([np.random.randint(0, 10) for _ in range(100)]),
        "float": pd.Series([np.random.rand() for _ in range(100)])
    })

    vars = MetaVar.detect(df)
    assert len(vars) == 2
    assert isinstance(vars, list)
    assert isinstance(vars[0], IntVar)
    assert isinstance(vars[1], FloatVar)
    