"""Module containing an interface to the faker package."""

from faker import Faker

from metasynth.distribution.base import BaseDistribution


class FakerDistribution(BaseDistribution):
    def __init__(self, faker_type, locale="en_US"):
        self.faker_type = faker_type
        self.locale = locale
        self.fake = Faker(locale=locale)

    @classmethod
    def _fit(cls, values, faker_type="city", locale="en_US"):
        return cls(faker_type, locale)

    def __str__(self):
        return f"Faker.{self.faker_type}"

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
        return name.startswith("faker") and len(name.split(".")) >= 2

    @classmethod
    def fit_kwargs(cls, name):
        split_name = name.split(".")
        if len(split_name) == 2:
            return {"faker_type": split_name[1]}
        return {"faker_type": split_name[1],
                "locale": split_name[2]}
