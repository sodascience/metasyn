"""Module containing an interface to the faker package."""
from typing import Iterable, Optional

from faker import Faker
from lingua import Language, LanguageDetectorBuilder
import numpy as np

from metasynth.distribution.base import metadist, BaseDistribution, UniqueDistributionMixin


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


@metadist(implements="core.unique_faker", var_type="string")
class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):
    """Faker distribution that returns unique values."""


@metadist(implements="core.unstructured", var_type="string")
class UnstructuredTextDistribution(BaseDistribution):
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

    def __init__(self, locale: str, avg_sentences: Optional[float], avg_words: int):
        self.locale: str = locale
        self.avg_sentences = avg_sentences
        self.avg_words = avg_words
        self.fake = Faker(locale=self.locale)

    @classmethod
    def _fit(cls, values):
        """Select the appropriate faker function and locale."""
        detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
        lang = detector.detect_language_of("\n".join(values))
        lang_str = str(lang.iso_code_639_1).split(".")[-1]

        try:
            Faker(lang_str)
        except AttributeError:
            lang_str = "EN"

        all_text = " ".join(values)
        n_non_empty = (values != "").sum()
        n_sentences = len(all_text.split("."))
        n_words = len(all_text.split(" "))
        if n_sentences < n_non_empty//3:
            avg_sentence = None
        else:
            avg_sentence = n_sentences/len(values)
        avg_words = n_words/len(values)
        return cls(lang_str, avg_sentence, avg_words)

    # def __str__(self):
        # return f"faker.{self.faker_type}.{self.locale}"

    def draw(self):
        if self.avg_sentences is None:
            n_words = np.random.randint(2*int(self.avg_words+1)) + 1
            sentence = self.fake.sentence(n_words)
            return sentence[:-1]

        n_sentences = np.random.randint(2*int(self.avg_sentences+1))
        n_words = np.random.randint(2*int(self.avg_words+1)+1)
        return " ".join(self.fake.sentence(nb_words=n_words) for _ in range(n_sentences))

    def information_criterion(self, values: Iterable) -> float:
        return 99999

    @classmethod
    def default_distribution(cls):
        return cls("en_US", 3, 6)

    def _param_dict(self):
        return {
            "locale": self.locale,
            "avg_sentences": self.avg_sentences,
            "avg_words": self.avg_words,
        }

    @classmethod
    def _param_schema(cls):
        return {
            "locale": {"type": "string"},
            "avg_sentences": {"type": ["number", "null"]},
            "avg_words": {"type": "number"},
        }
