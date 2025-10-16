"""Test the string type distribution inference."""
import pandas as pd
import polars as pl
import pytest
from pytest import mark

from metasyn.distribution.faker import FakerFitter, UniqueFakerDistribution
from metasyn.distribution.freetext import FreeTextFitter
from metasyn.distribution.regex import UniqueRegexDistribution
from metasyn.privacy import BasicPrivacy
from metasyn.var import MetaVar


@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_faker(series_type):
    """Test the faker distribution."""
    dist = FakerFitter(BasicPrivacy()).fit(series_type([1, 2, 3]))
    assert isinstance(dist.to_dict(), dict)
    assert isinstance(dist.draw(), str)
    assert 'faker' in str(dist)
    assert dist.locale == "en_US"
    assert dist.faker_type == "city"
    var = MetaVar("some_city", "string", dist, prop_missing=0.0)
    series_1 = var.draw_series(100, seed=1234)
    series_2 = var.draw_series(100, seed=1234)
    assert all(series_1 == series_2)


@mark.parametrize(
    "series,lang,avg_sentences,avg_words", [
            (pl.Series(
                ["hotdog", "mother", "yes", "tip", "crate",
                 "sink", "dark", "crossbar", "toilet", "grow", "patient"]), "EN", None, 1),
            (pl.Series(["hotdog mother", "yes tip", "crate sink", "dark crossbar", "toilet grow",
                        "patient is"]), "EN", None, 2),
            (pl.Series(["hotdog mother.", "yes tip.", "crate sink.", "dark crossbar.",
                        "toilet grow.", "patient is."]), "EN", 1, 2),
            (pl.Series(["gaat naar school. Ik ben benieuwd.", "Wat is vraag? Wanneer komt hij."]),
             "NL", 2, 6)
    ])
def test_free_text(series, lang, avg_sentences, avg_words):
    """Test whether the free text distribution does the right inference."""
    dist = FreeTextFitter(BasicPrivacy()).fit(series)
    assert dist.locale == lang
    assert dist.avg_sentences == avg_sentences
    assert dist.avg_words == avg_words
    var = MetaVar("some_var", "string", dist, prop_missing=0.0)
    series_1 = var.draw_series(100, seed=1234)
    series_2 = var.draw_series(100, seed=1234)
    assert all(series_1 == series_2)


def test_unique_regex():
    dist = UniqueRegexDistribution(r"[0-9]")
    var = MetaVar("some_var", "string", dist, prop_missing=0.0)

    series = var.draw_series(10, None)
    assert len(series.unique()) == 10

    with pytest.raises(ValueError):
        var.draw()

def test_unique_faker():
    dist = UniqueFakerDistribution("city")
    var = MetaVar("some_var", "string", dist, prop_missing=0.0)

    series = var.draw_series(1000, None)
    assert len(series.unique()) == 1000
