"""Module all string distributions."""
from __future__ import annotations

from typing import Iterable, Optional, Union

# from lingua._constant import LETTERS, PUNCTUATION
import regex
from faker import Faker
from lingua import LanguageDetectorBuilder  # pylint: disable=no-name-in-module
from regexmodel import NotFittedError, RegexModel
from scipy.stats import poisson

from metasyn.distribution.base import (
    BaseConstantDistribution,
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


@metadist(implements="core.faker", var_type="string")
class UniqueFakerDistribution(UniqueDistributionMixin, FakerDistribution):
    """Faker distribution that returns unique values.

    See :class:`~FakerDistribution` for examples and explanation.
    """


@metadist(implements="core.freetext", var_type="string")
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

    @classmethod
    def _fit(cls, values, max_values: int = 50):
        """Select the appropriate faker function and locale."""
        lang_str = cls.detect_language(values[:max_values])
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
        series = self._to_series(values)
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


@metadist(implements="core.constant", var_type="string")
class StringConstantDistribution(BaseConstantDistribution):
    """Constant string distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: str
        Value that will be returned when drawn.

    Examples
    --------
    >>> ConstantDistribution("some_string")
    """

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls("text")

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "string"}
        }

@metadist(implements="core.regex", var_type="string", version="2.0")
class RegexDistribution(BaseDistribution):
    """Structured string distribution using regex.

    Main implementation details in the regexmodel package:
    https://github.com/sodascience/regexmodel

    This distribution tries to create a regex that fits the variable. This
    regex also contains statistical information about the probabilities if the
    regex has multiple options (e.g. a|b). The regex is only a subset what is provided
    by the python re package. What is currently implemented:

    - Parentheses with multiple options and no modifiers, e.g. ([a]|[b]|[c])
    - Square brackets without negation, e.g. [abc]
    - Ranges [A-Z], [a-z], [0-9], but not subranges (e.g. [0-3])
    - Repetition quantifiers (curly brackets) with minimum and maximum [A-Z]{3,6}, but not [A-Z]{6}.

    When fitting the RegexDistribution using the fit method, pay attention to the
    count_thres and method parameters. By default these will be dynamic and take reasonable
    values for the input, but in some cases it can be important to set them manually.
    The count_thres parameter sets the minimum number of times a regex element needs to
    be used. So, if count_thres=2, and there is only one value starting with "a", then the regex
    will never start with "a". In effect, a higher value will provide more privacy, less
    utility and a faster fit. The other parameter "method" has a small effect on the accuracy
    of the regex, and a larger effect on the worst case time consumption for fitting.
    Set to "accurate" for the best result, and "fast" for the fastest result.

    Examples that this distribution should work reasonably for are: email,
    ID's, telephone numbers, ip addresses, etc.

    Parameters
    ----------
    regex_data:
        Valid inputs for the regex model are:
        - str: String with a regex (that falls within the specifications, see above).
        - dict: Serialized version of the regex model, as it is coming from a JSON file.
        - RegexModel: Initialized regex model.

    Examples
    --------
    >>> RegexDistribution(r"AB[0-9]{4}").draw()
    "AB8123"
    >>> RegexDistribution(r"(a|b|c)10)").draw()
    "b10"
    """

    def __init__(self, regex_data: Union[str, dict, RegexModel]):
        self.regex_model = RegexModel(regex_data)

    @classmethod
    def _fit(cls, values, count_thres: Optional[int] = None, method: str = "auto"):
        """Fit a regex to structured strings.

        Arguments
        ---------
        values:
            Values to be fitted (pl.Series).
        count_thres:
            Threshold for regex elements, so that a regex element can only be used if
            the number of values satisfying said element is higher than the threshold.
        method:
            Method for fitting the regex model. Possible values are ["accurate", "fast", "auto"]
            The "auto" method switches between the "accurate" and "fast" methods depending on
            the number of characters (fast if #char > 10000) in the series.
        """
        if method == "auto":
            if values.str.len_chars().mean() > 10:
                method = "fast"
            else:
                method = "accurate"

        # Make count_thres ~= #values/100 up to 50 if in auto mode.
        if count_thres is None:
            count_thres = min(50, max(2, round(len(values)/50)))

        # Try to fit the values, if it cannot be fit, then use the default distribution.
        try:
            model = RegexModel.fit(values, count_thres=count_thres, method=method)
        except NotFittedError:
            return cls.default_distribution()
        return cls(model)

    def draw(self):
        return self.regex_model.draw()

    def _param_dict(self):
        return {"regex_data": self.regex_model.serialize()}

    @property
    def _params_formatted(self):
        return f"\t- regex: {self.regex_model.regex}"

    @classmethod
    def _param_schema(cls):
        return {
            "regex": {
                "type": "string"
            },
            "counts": {
                "type": "array"
            }
        }

    @classmethod
    def default_distribution(cls):
        return cls(r"[ABC][0-9]{3,4}")


@metadist(implements="core.regex", var_type="string")
class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):
    """Unique variant of the regex distribution.

    See :class:`~RegexDistribution` for examples and explanation on this distribution.
    """
