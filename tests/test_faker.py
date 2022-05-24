import pandas as pd

from metasynth.distribution.faker import FakerDistribution


def test_faker():
    var = FakerDistribution.fit(pd.Series([1, 2, 3]))
    assert isinstance(var.to_dict(), dict)
    assert isinstance(var.draw(), str)
    assert var.is_named("faker.city.nl_NL")
    fit_kwargs = var.fit_kwargs("faker.city.nl_NL")
    assert fit_kwargs["faker_type"] == "city"
    assert fit_kwargs["locale"] == "nl_NL"
    fit_kwargs = var.fit_kwargs("faker.city")
    assert fit_kwargs["faker_type"] == "city"
    assert "locale" not in fit_kwargs
    assert str(var).startswith("Faker")
