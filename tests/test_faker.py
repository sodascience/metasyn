import pandas as pd
import polars as pl
from pytest import mark

from metasyn.distribution.faker import FakerDistribution


@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_faker(series_type):
    var = FakerDistribution.fit(series_type([1, 2, 3]))
    assert isinstance(var.to_dict(), dict)
    assert isinstance(var.draw(), str)
    assert 'faker' in str(var)
    assert var.locale == "en_US"
    assert var.faker_type == "city"
