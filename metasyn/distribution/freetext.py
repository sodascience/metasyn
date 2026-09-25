"""Module for the free text distribution that generates random words after on another."""
from __future__ import annotations

from typing import Optional

import regex
from faker import Faker
from scipy.stats import poisson

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    builtin_fitter,
    convert_to_series,
    metadist,
)

LETTERS = regex.compile(r"\p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}|\p{L}+")
PUNCTUATION = regex.compile(r"\p{P}")


@metadist(name="core.freetext", var_type="string")
class FreeTextDistribution(BaseDistribution):
    """Free text distribution.

    This distribution generates random sentences using
    the Faker package. The average number of sentences and words per item
    are detected using regexes.

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

    def __init__(self, locale: str, avg_sentences: float, avg_words: float):
        self.locale: str = locale
        self.avg_sentences = avg_sentences
        self.avg_words: float = avg_words
        self.fake = Faker(locale=self.locale)


    def draw(self):
        if self.avg_sentences is None or self.avg_sentences < 0.01:
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
            return 10 + (avg_chars - self.avg_words*6)**2  # type: ignore  # Workaround polars typing
        return 99999999

    @classmethod
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
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

@builtin_fitter(distribution=FreeTextDistribution, var_type="string")
class FreeTextFitter(BaseFitter):
    """Fitter for the freetext distribution."""

    distribution: type[FreeTextDistribution]

    def __init__(self, privacy, lang_str: Optional[str] = None):
        self.lang_str = lang_str
        super().__init__(privacy)

    def _fit(self, series, fit_log):
        """Select the appropriate faker function and locale."""
        lang_str = self.lang_str
        if self.lang_str is None:
            fit_log.add(method="Language is not detected anymore since metasyn version 3.0, "
                        "using English. You can manually set the distribution to another language:"
                        " FreeTextDistribution(locale='nl_NL').")
            lang_str = "en_US"

        try:
            Faker(lang_str)
        except AttributeError:
            fit_log.add(method=f"Language '{lang_str}' is not supported by the Faker package, "
                        "using 'en_US'.")
            lang_str = "en_US"

        all_text = "\n".join(series)
        n_non_empty = (series != "").sum()
        n_punctuation = len(list(PUNCTUATION.finditer(all_text)))
        n_words = len(list(LETTERS.finditer(all_text)))
        if n_punctuation < n_non_empty//3:
            avg_sentence = 0.0
        else:
            avg_sentence = n_punctuation/len(series)
        avg_words = n_words/len(series)
        fit_log.add(method="Using average number of sentences and words as parameters.")
        return self.distribution(lang_str, avg_sentence, avg_words)
