"""Module all string distributions."""
from __future__ import annotations

from typing import Iterable, Optional

# from lingua._constant import LETTERS, PUNCTUATION
import regex
from faker import Faker
from lingua import LanguageDetectorBuilder
from scipy.stats import poisson

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    metadist,
    metafit,
    convert_to_series,
)

LETTERS = regex.compile(r"\p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}|\p{L}+")
PUNCTUATION = regex.compile(r"\p{P}")


@metadist(name="core.freetext", var_type="string")
class FreeTextDistribution(BaseDistribution):
    """Free text distribution.

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

    def __init__(self, locale: str, avg_sentences: Optional[float], avg_words: float):
        self.locale: str = locale
        self.avg_sentences = avg_sentences
        self.avg_words = avg_words
        self.fake = Faker(locale=self.locale)


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
        series = convert_to_series(values)
        # Check the average number of characters
        avg_chars = series.str.len_chars().mean()
        if avg_chars is not None and avg_chars >= 25:  # type: ignore  # Workaround polars typing
            lang = self.detect_language(series)
            if lang is not None:
                return -1.0
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

@metafit(distribution=FreeTextDistribution, var_type="string")
class FreeTextFitter(BaseFitter):
    """Fitter for the freetext distribution."""

    def _fit(self, series, max_values: int = 50):
        """Select the appropriate faker function and locale."""
        lang_str = self.detect_language(series[:max_values])
        if lang_str is None:
            return self.distribution.default_distribution()

        try:
            Faker(lang_str)
        except AttributeError:
            lang_str = "EN"

        all_text = "\n".join(series)
        n_non_empty = (series != "").sum()
        n_punctuation = len(list(PUNCTUATION.finditer(all_text)))
        n_words = len(list(LETTERS.finditer(all_text)))
        if n_punctuation < n_non_empty//3:
            avg_sentence = None
        else:
            avg_sentence = n_punctuation/len(series)
        avg_words = n_words/len(series)
        return self.distribution(lang_str, avg_sentence, avg_words)

    def detect_language(self, values: Iterable) -> Optional[str]:
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
