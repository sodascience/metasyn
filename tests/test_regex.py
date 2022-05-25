import pandas as pd
from metasynth.distribution.string import RegexDistribution, SingleRegex,\
    DigitRegex, AlphaNumericRegex, LettersRegex, LowercaseRegex, UppercaseRegex
from random import choice
import string
from pytest import mark
import numpy as np


def test_regex_single_digit():
    series = pd.Series(["R123", "R823", "R124"])
    dist = RegexDistribution.fit(series)
    assert len(dist.re_list) == 2
    assert isinstance(dist.re_list[0], SingleRegex)
    assert isinstance(dist.re_list[1], DigitRegex)
    assert dist.re_list[0].character_selection == ["R"]
    assert dist.re_list[1].min_digit == 3
    assert dist.re_list[1].max_digit == 3
    assert str(dist) == r"[R]\d{3,3}"

    for draw_str in [dist.draw() for _ in range(10)]:
        assert len(draw_str) == 4
        assert draw_str[0] == "R"


@mark.parametrize(
    "digit_set,dist_class,regex_str",
    [
        (string.ascii_letters+string.digits, AlphaNumericRegex, r"\w{10,10}"),
        (string.ascii_letters, LettersRegex, r"[a-zA-Z]{10,10}"),
        (string.digits, DigitRegex, r"\d{10,10}"),
        (string.ascii_lowercase, LowercaseRegex, r"[a-z]{10,10}"),
        (string.ascii_uppercase, UppercaseRegex, r"[A-Z]{10,10}"),
    ]
)
def test_digits(digit_set, dist_class, regex_str):
    def draw():
        draw_str = ""
        for _ in range(10):
            draw_str += choice(digit_set)
        return draw_str

    series = pd.Series([draw() for _ in range(100)])
    dist = RegexDistribution.fit(series)
    assert len(dist.re_list) == 1
    assert isinstance(dist.re_list[0], dist_class)
    assert dist.re_list[0].min_digit == 10
    assert dist.re_list[0].max_digit == 10
    assert dist.to_dict()["parameters"]["re_list"][0] == regex_str
    assert np.all([len(dist.draw()) == 10 for _ in range(100)])
    assert np.all([c in digit_set for c in dist.draw()])


def test_regex_alpha():
    def draw():
        draw_str = ""
        for _ in range(10):
            draw_str += choice(string.ascii_letters+string.digits)
        return draw_str

    series = pd.Series([draw() for _ in range(100)])
    print(series)
    dist = RegexDistribution.fit(series)
    assert len(dist.re_list) == 1
    assert isinstance(dist.re_list[0], AlphaNumericRegex)
    assert dist.re_list[0].min_digit == 10
    assert dist.re_list[0].max_digit == 10
    assert dist.to_dict()["parameters"]["re_list"][0] == r"\w{10,10}"

def test_regex_lower():
    pass