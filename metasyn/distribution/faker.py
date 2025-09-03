"""Module all string distributions."""
from __future__ import annotations

from typing import Iterable, Optional, Union

# from lingua._constant import LETTERS, PUNCTUATION
import regex
from faker import Faker
from lingua import LanguageDetectorBuilder
from regexmodel import NotFittedError, RegexModel
from scipy.stats import poisson

from metasyn.distribution.base import (
    BaseDistribution,
    UniqueDistributionMixin,
    metadist,
)

LETTERS = regex.compile(r"\p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}|\p{L}+")
PUNCTUATION = regex.compile(r"\p{P}")

@metadist(implements="core.faker", var_type="string")
class FakerDistribution(BaseDistribution):
    """Faker distribution for cities, addresses, etc.

    This is mainly an interface for the faker package, so that it
    can be used within the metasyn package. It doesn't have any
    true fitting/statistical inference method, so it has to be manually
    selected.

    Parameters
    ----------
    faker_type: str
        The provider function in the faker package, e.g. 'city' or 'ipv4', etc.
    locale: str
        Locale used for the faker package.

    Examples
    --------
    >>> FakerDistribution(faker_type="city", locale="en_US")
    >>> FakerDistribution(faker_type="address", locale="nl_NL")

    """

    def __init__(self, faker_type: str, locale: str = "en_US"):
        self.faker_type: str = faker_type
        self.locale: str = locale
        self.fake: Faker = Faker(locale=locale)

    @classmethod
    def _fit(cls, values, faker_type: str = "city", locale: str = "en_US"): # noqa: ARG003
        """Select the appropriate faker function and locale."""
        return cls(faker_type, locale)

    def draw(self):
        return getattr(self.fake, self.faker_type)()

    def information_criterion(self, values: Iterable) -> float: # noqa: ARG002
        return 99999

    @classmethod
    def default_distribution(cls):
        return cls("city")

    def _param_dict(self):
        return {
            "faker_type": self.faker_type,
            "locale": self.locale
        }

    @classmethod
    def _param_schema(cls):
        return {
            "faker_type": {"type": "string"},
            "locale": {"type": "string"},
        }


@metadist(implements="core.faker", var_type="string")
class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):
    """Faker distribution that returns unique values.

    See :class:`~FakerDistribution` for examples and explanation.
    """

