"""Module containing an interface to the faker package."""
from typing import Iterable

from faker import Faker

from metasynth.distribution.base import metadist, BaseDistribution


@metadist(implements="core.faker", var_type="string")
class FakerDistribution(BaseDistribution):
    """Distribution for the faker package.

    This is mainly an interface for the faker package, so that it
    can be used within the MetaSynth package. It doesn't have any
    true fitting/statistical inference method, so it has to be manually
    selected.

    Parameters
    ----------
    faker_type: str
        The provider function in the faker package, e.g. 'city' or 'ipv4', etc.
    locale: str
        Locale used for the faker package.
    """

    def __init__(self, faker_type: str, locale: str = "en_US"):
        self.faker_type: str = faker_type
        self.locale: str = locale
        self.fake: Faker = Faker(locale=locale)

    @classmethod
    def _fit(cls, values, faker_type: str = "city", locale: str = "en_US"):  \
            # pylint: disable=arguments-differ
        """Select the appropriate faker function and locale."""
        return cls(faker_type, locale)

    def __str__(self):
        return f"faker.{self.faker_type}.{self.locale}"

    def draw(self):
        return getattr(self.fake, self.faker_type)()

    def information_criterion(self, values: Iterable) -> float:
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


@metadist(implements="core.unique_faker", var_type="string", unique=True)
class UniqueFakerDistribution(FakerDistribution):
    """Faker distribution that returns unique values.

    It will raise a ValueError if it runs out of new values to draw from

    Parameters
    ----------
    faker_type: str
        The provider function in the faker package, e.g. 'city' or 'ipv4', etc.
    locale: str
        Locale used for the faker package.
    """
    def __init__(self, faker_type: str, locale: str = "en_US"):
        super().__init__(faker_type, locale)
        self.key_set: set[str] = set()

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> str:
        n_try = 0
        while n_try < 1e5:
            new_val = super().draw()
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val
            n_try += 1
        raise ValueError("Failed to draw unique string after 100.000 tries.")

    def information_criterion(self, values):
        return 99999
