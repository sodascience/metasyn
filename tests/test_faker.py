import pandas as pd
import polars as pl
from pytest import mark

from metasyn.distribution.string import FakerDistribution, FreeTextDistribution


@mark.parametrize("series_type", [pd.Series, pl.Series])
def test_faker(series_type):
    var = FakerDistribution.fit(series_type([1, 2, 3]))
    assert isinstance(var.to_dict(), dict)
    assert isinstance(var.draw(), str)
    assert 'faker' in str(var)
    assert var.locale == "en_US"
    assert var.faker_type == "city"


@mark.parametrize(
    "series,lang,avg_sentences,avg_words", [
            (pl.Series(
                ["hotdog", "mother", "yes", "tip", "crate",
                 "sink", "dark", "crossbar", "toilet", "grow", "patient"]), "EN", None, 1),
            (pl.Series(["hotdog mother", "yes tip", "crate sink", "dark crossbar", "toilet grow",
                        "patient is"]), "EN", None, 2),
            (pl.Series(["hotdog mother.", "yes tip.", "crate sink.", "dark crossbar.", "toilet grow.",
                        "patient is."]), "EN", 1, 2),
            (pl.Series(["gaat naar school. Ik ben benieuwd.", "Wat is vraag? Wanneer komt hij."]),
             "NL", 2, 6)
    ])
def test_free_text(series, lang, avg_sentences, avg_words):
    dist = FreeTextDistribution.fit(series)
    assert dist.locale == lang
    assert dist.avg_sentences == avg_sentences
    assert dist.avg_words == avg_words
    dist.draw()
