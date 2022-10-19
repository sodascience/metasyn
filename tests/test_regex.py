import string
from random import choice

import numpy as np
import pandas as pd
from pytest import mark, raises

from metasynth.distribution.regex import RegexDistribution, UniqueRegexDistribution
from metasynth.distribution.regex.element import SingleRegex,\
    DigitRegex, AlphaNumericRegex, LettersRegex, LowercaseRegex, UppercaseRegex,\
    AnyRegex


def test_regex_single_digit():
    series = pd.Series(["R123", "R837", "R354", "R456", "R578", "R699"])
    dist = RegexDistribution.fit(series)
    dist_unique = UniqueRegexDistribution.fit(series)
    print(dist.to_dict(), dist_unique.to_dict())
    assert dist.information_criterion(series) < dist_unique.information_criterion(series)

    def check_regex_dist(dist):
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

    check_regex_dist(dist)

    re_string_list = dist.to_dict()["parameters"]["re_list"]
    new_dist = RegexDistribution(re_string_list)
    check_regex_dist(new_dist)


def test_regex_unique():
    series = pd.Series(["R1", "R2", "R3", "R4", "R5", "R6"])
    dist = UniqueRegexDistribution.fit(series)
    values = [dist.draw() for _ in range(10)]
    assert len(set(values)) == 10
    assert set(values) == set(["R" + x for x in string.digits])
    with raises(ValueError):
        dist.draw()
    dist.draw_reset()
    dist.draw()


@mark.parametrize(
    "dist_class,regex_str,regex_str_alt",
    [
        (AlphaNumericRegex, r"\w{1,1}", r"\w"),
        (LettersRegex, r"[a-zA-Z]{1,1}",  r"[a-zA-Z]"),
        (LowercaseRegex, r"[a-z]{1,1}", r"[a-z]"),
        (UppercaseRegex, r"[A-Z]{1,1}", r"[A-Z]"),
        (DigitRegex, r"\d{1,1}", r"\d"),
        (AnyRegex, r".[]{1,1}", r".[]"),
    ]
)
def test_optional_length(dist_class, regex_str, regex_str_alt):
    dist, _ = dist_class.from_string(regex_str)
    dist_alt, _ = dist_class.from_string(regex_str_alt)
    assert str(dist) == str(dist_alt)


@mark.parametrize(
    "digit_set,dist_class,regex_str,n_digits",
    [
        (string.ascii_letters+string.digits, AlphaNumericRegex, r"\w{10,10}", 10),
        (string.ascii_letters, LettersRegex, r"[a-zA-Z]{10,10}", 10),
        (string.digits, DigitRegex, r"\d{10,10}", 10),
        (string.ascii_lowercase, LowercaseRegex, r"[a-z]{10,10}", 10),
        (string.ascii_uppercase, UppercaseRegex, r"[A-Z]{10,10}", 10),
    ]
)
def test_digits(digit_set, dist_class, regex_str, n_digits):
    def draw():
        draw_str = ""
        for _ in range(n_digits):
            draw_str += choice(digit_set)
        return draw_str

    series = pd.Series([draw() for _ in range(100)])
    dist = RegexDistribution.fit(series)
    assert len(dist.re_list) == 1
    assert isinstance(dist.re_list[0], dist_class)
    assert dist.re_list[0].min_digit == n_digits
    assert dist.re_list[0].max_digit == n_digits
    assert dist.to_dict()["parameters"]["re_list"][0][0] == regex_str
    assert np.all([len(dist.draw()) == n_digits for _ in range(100)])
    assert np.all([c in digit_set for c in dist.draw()])
    new_dist = dist_class.from_string(str(dist))[0]
    assert isinstance(new_dist, dist_class)
    assert new_dist.min_digit == dist.re_list[0].min_digit
    assert new_dist.max_digit == dist.re_list[0].max_digit
