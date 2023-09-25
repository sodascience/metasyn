"""Module containing an interface to the faker package."""
from typing import Iterable, Optional

from faker import Faker
from lingua import LanguageDetectorBuilder
from lingua._constant import LETTERS, PUNCTUATION
# LETTERS: Pattern = regex.compile(r"\p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}|\p{L}+")
# PUNCTUATION: Pattern = regex.compile(r"\p{P}")
from scipy.stats import poisson

from metasyn.distribution.base import metadist, BaseDistribution, UniqueDistributionMixin


@metadist(implements="core.faker", var_type="string")
class FakerDistribution(BaseDistribution):
    """Distribution for the faker package.

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


@metadist(implements="core.freetext", var_type="string")
class FreeTextDistribution(BaseDistribution):
    """Distribution for free text.

    This distribution detects the language and generates sentences using
    the Faker package. The average number of sentences and words per item
    are detected using regexes (with the lingua package).

    Parameters
    ----------
    locale: str
        Locale used for the faker package.
    avg_sentences:
        Average number of sentences (punctuation marks) per (non-NA) row,
        if None do not make sentences.
    avg_words:
        Average number of words per (non-NA) row.
    """

    def __init__(self, locale: str, avg_sentences: Optional[float], avg_words: int):
        self.locale: str = locale
        self.avg_sentences = avg_sentences
        self.avg_words = avg_words
        self.fake = Faker(locale=self.locale)

    @classmethod
    def _fit(cls, values):
        """Select the appropriate faker function and locale."""
        lang_str = cls.detect_language(values)
        if lang_str is None:
            return cls.default_distribution()

        try:
            Faker(lang_str)
        except AttributeError:
            lang_str = "EN"

        all_text = "\n".join(values)
        n_non_empty = (values != "").sum()
        n_punctuation = len(list(PUNCTUATION.finditer(all_text)))
        n_words = len(list(LETTERS.finditer(all_text)))
        if n_punctuation < n_non_empty//3:
            avg_sentence = None
        else:
            avg_sentence = n_punctuation/len(values)
        avg_words = n_words/len(values)
        return cls(lang_str, avg_sentence, avg_words)

    @classmethod
    def detect_language(cls, values: Iterable) -> Optional[str]:
        """Detect the language of some text.

        Parameters
        ----------
        values:
            Values to detect the language of (usually polars dataframe).

        Returns
        -------
        language:
            Two letter ISO code to represent the language, or None if it could not be determined.
        """
        detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
        lang = detector.detect_language_of("\n".join(values))
        if lang is None:
            return None
        return str(lang.iso_code_639_1).rsplit(".", maxsplit=1)[-1]

    def draw(self):
        if self.avg_sentences is None:
            n_words = max(1, poisson(self.avg_words).rvs())
            sentence = self.fake.sentence(n_words)
            return sentence[:-1]

        n_sentences = max(1, poisson(self.avg_sentences).rvs())
        avg_words_per_sent = max(1, self.avg_words/max(1, self.avg_sentences))
        n_words = max(1, poisson(avg_words_per_sent).rvs())
        return " ".join(self.fake.sentence(nb_words=n_words) for _ in range(n_sentences))

    def information_criterion(self, values) -> float:
        # series = self._to_series(values)
        # lang = self.detect_language(series)
        # if lang is None:
        # Don't use this distribution by default (for now).
        return 99999999

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
