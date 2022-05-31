"""Module containing an interface to the faker package."""

from faker import Faker

from metasynth.distribution.base import BaseDistribution


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
    def __init__(self, faker_type, locale="en_US"):
        self.faker_type = faker_type
        self.locale = locale
        self.fake = Faker(locale=locale)

    @classmethod
    def _fit(cls, values, faker_type="city", locale="en_US"):  # pylint: disable=arguments-differ
        """Select the appropriate faker function and locale."""
        return cls(faker_type, locale)

    def __str__(self):
        return f"faker.{self.faker_type}.{self.locale}"

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": {
                "faker_type": self.faker_type,
                "locale": self.locale
            }
        }

    def draw(self):
        return getattr(self.fake, self.faker_type)()

    @classmethod
    def is_named(cls, name):
        if name == cls.__name__:
            return True
        return name.startswith("faker") and len(name.split(".")) >= 2

    @classmethod
    def fit_kwargs(cls, name):
        if name == cls.__name__:
            return {}
        split_name = name.split(".")
        if len(split_name) == 2:
            return {"faker_type": split_name[1]}
        return {"faker_type": split_name[1],
                "locale": split_name[2]}
