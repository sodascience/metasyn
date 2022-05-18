import pandas as pd
import numpy as np
from metasynth.var import MetaVar, CategoricalVar, IntVar, FloatVar, StringVar
from metasynth.distribution import CatFreqDistribution, NormalDistribution,\
    StringFreqDistribution
from metasynth.distribution import DiscreteUniformDistribution
from metasynth.distribution import UniformDistribution


def check_var(series, var_type, dist_class):
    assert isinstance(series, pd.Series)
    var = MetaVar.detect(series)
    var.fit()
    new_series = var.draw_series(100)
    assert len(series) == len(new_series)
    assert new_series.dtype == series.dtype
    assert isinstance(var, var_type)
    assert isinstance(var.distribution, dist_class)
    return new_series


def test_categorical():
    series = pd.Series(np.random.choice(["a", "b", "c"], size=100), dtype="category")
    new_series = check_var(series, CategoricalVar, CatFreqDistribution)
    assert set(np.unique(series)) == set(np.unique(new_series))


def test_integer():
    series = pd.Series([np.random.randint(0, 10) for _ in range(100)], dtype="int8")
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
    series = pd.Series(np.random.choice(["a", "b", "c"], size=100))
    new_series = check_var(series, StringVar, StringFreqDistribution)
    assert set(np.unique(series)) == set(np.unique(new_series))
