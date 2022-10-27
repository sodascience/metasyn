import pandas as pd
import polars as pl
from pytest import mark

from metasynth.distribution.faker import FakerDistribution


@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_faker(series_type):
    var = FakerDistribution.fit(series_type([1, 2, 3]))
    assert isinstance(var.to_dict(), dict)
    assert isinstance(var.draw(), str)
    assert var.is_named("faker.city.nl_NL")
    fit_kwargs = var.fit_kwargs("faker.city.nl_NL")
    assert fit_kwargs["faker_type"] == "city"
    assert fit_kwargs["locale"] == "nl_NL"
    fit_kwargs = var.fit_kwargs("faker.city")
    assert fit_kwargs["faker_type"] == "city"
    assert "locale" not in fit_kwargs
    assert str(var).startswith("faker")
