import pandas as pd
from metasynth.distribution.string import RegexDistribution, SingleRegex,\
    DigitRegex, AlphaNumericRegex, LettersRegex, LowercaseRegex, UppercaseRegex,\
    UniqueRegexDistribution
from random import choice
import string
from pytest import mark, raises
import numpy as np


def test_regex_single_digit():
    series = pd.Series(["R123", "R823", "R124"])
    dist = RegexDistribution.fit(series)
    dist_unique = UniqueRegexDistribution.fit(series)
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
    dist_non_unique = RegexDistribution.fit(series)
    assert dist.information_criterion(series) < dist_non_unique.information_criterion(series)
    values = [dist.draw() for _ in range(10)]
    assert len(set(values)) == 10
    assert set(values) == set(["R" + x for x in string.digits])
    with raises(ValueError):
        dist.draw()
    dist.draw_reset()
    dist.draw()


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
    new_dist = dist_class.from_string(str(dist))
    assert isinstance(new_dist, dist_class)
    assert new_dist.min_digit == dist.re_list[0].min_digit
    assert new_dist.max_digit == dist.re_list[0].max_digit
