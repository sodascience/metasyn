"""Module containing string distributions."""

from abc import abstractmethod, ABC
from copy import deepcopy
import re
import random
import string

import numpy as np

from metasynth.distribution.base import BaseDistribution


def _get_prop_solved(values, new_values):
    """Get the proportion of the characters that were resolved with the regex."""
    n_char = np.sum([len(val) for val in values])
    n_new_char = np.sum([len(val) for val in new_values])
    return (n_char-n_new_char)/n_char


class RegexDistribution(BaseDistribution):
    """Distribution that uses a strategy similar to regex.

    The idea behind this method is that for structured strings
    regexes works very well. It uses a greedy algorithm to build the
    regex. It does not implement all regexes, nor does it claim to be
    compliant with standard ones.

    Parameters
    ----------
    re_list: list of BaseRegexElement
        List of basic regex elements in the order that they occur.
    """

    aliases = ["regex"]

    def __init__(self, re_list):
        self.re_list = re_list

    @classmethod
    def _fit(cls, values):
        n_char = np.sum([len(val) for val in values])
        if n_char == 0:
            return cls([])
        best_solution = None
        regex_classes = [
            DigitRegex, AlphaNumericRegex, LowercaseRegex, UppercaseRegex,
            LettersRegex, SingleRegex]

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, regexer = re_class.fit(values)
            prop_solved = _get_prop_solved(values, new_values)
            cur_ic = regexer.information_criterion(prop_solved)
            if best_solution is None or cur_ic > best_solution["ic"]:
                best_solution = {
                    "ic": cur_ic,
                    "re": regexer,
                    "values": new_values,
                }
        return cls([best_solution["re"]]) + cls._fit(best_solution["values"])

    def __add__(self, other):
        return RegexDistribution(self.re_list + other.re_list)

    def draw(self):
        cur_str = ""
        for rex in self.re_list:
            cur_str += rex.draw()
        return cur_str

    def __str__(self):
        return "".join([str(x) for x in self.re_list])

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": {
                    "re_list": [str(x) for x in self.re_list]
                }
        }


class BaseRegexElement(ABC):
    """Base class for regex elements."""

    def information_criterion(self, prop_solved):
        """Compute the information criterion for iterative algorithm.

        It is similar to the Akaike Information Criterion in the sense that
        it takes into account the number of parameters and the log likelihood of
        the values. In this case we are trying to estimate the addition to the IC
        due to the newly proposed regex element. Since the values are "solved/regex'ed"
        sequentially, the more of the values are regex'ed, the more likely the regex will be/
        the fewer new regexes need to be added. If each of the regexes has the same efficiency
        then the added IC would be Total IC * proportion solved. We use this mean field
        estimate to estimate which regex gives the best results.

        Parameters
        ----------
        prop_solved: float
            Proportion of the values that has been solved since the last iteration.

        Returns
        -------
        float:
            Estimate of remaining IC.
        """
        ic_add = self.n_param + self.log_options
        return -ic_add/(prop_solved+1e-8)

    @property
    def n_param(self):
        """int: The number of 'parameters' of the regex element"""
        return 1

    @property
    @abstractmethod
    def log_options(self):
        """float: The natural logarithm of the number of possibilities."""

    @classmethod
    @abstractmethod
    def fit(cls, values):
        """Match the regex to the values, using as many positions as possible.

        Parameters
        ----------
        values: array_like of str
            Remaining values to match the regex to.

        Returns
        -------
        BaseRegexElement:
            The optimized regex.
        """

    @abstractmethod
    def draw(self):
        """Draw a random string from the regex element.

        Returns
        -------
        str:
            The string drawn from the regex distribution element.
        """


class BaseRegexClass(BaseRegexElement):
    """Base class for repeating regex elements.

    Repeating elements are e.g. digits, letter, alphanumeric characters, etc.

    Parameters
    ----------
    min_digit: int
        Minimum number of repeats of the element.
    max_digit: int
        Maximum number of repeats of the element.
    """
    regex_str = None

    def __init__(self, min_digit, max_digit):
        self.min_digit = min_digit
        self.max_digit = max_digit

    @classmethod
    def fit(cls, values):
        regex = re.compile(cls.regex_str)
        new_values = []

        min_match = 99999
        max_match = 0
        for val in values:
            match = regex.search(val)
            if match is None:
                n_match = 0
            else:
                n_match = match.span()[1]
            min_match = min(n_match, min_match)
            max_match = max(n_match, max_match)
            new_values.append(val[n_match:])
        return new_values, cls(min_match, max_match)


class DigitRegex(BaseRegexClass):
    """Regex element for repeating digits."""
    regex_str = r"^\d{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(10)

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([str(np.random.randint(0, 10)) for _ in range(n_digit)])

    def __str__(self):
        return r"\d{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class AlphaNumericRegex(BaseRegexClass):
    """Regex element for repeating alphanumeric character."""
    regex_str = r"^\w{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters+string.digits))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters+string.digits) for _ in range(n_digit)])

    def __str__(self):
        return r"\w{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class LettersRegex(BaseRegexClass):
    """Regex element for repeating letters."""
    regex_str = r"^[a-zA-Z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters) for _ in range(n_digit)])

    def __str__(self):
        return "[a-zA-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class LowercaseRegex(BaseRegexClass):
    """Regex element for repeating lowercase letters."""
    regex_str = r"^[a-z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_lowercase))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(n_digit)])

    def __str__(self):
        return "[a-z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class UppercaseRegex(BaseRegexClass):
    """Regex element for repeating uppercase letters."""
    regex_str = r"^[A-Z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_uppercase))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_uppercase) for _ in range(n_digit)])

    def __str__(self):
        return "[A-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class SingleRegex(BaseRegexElement):
    """Regex element that is a choice on a set of (any) characters.

    Parameters
    ----------
    character_selection:
        The selection of characters available. An empty character ''
        is also allowed.
    """
    def __init__(self, character_selection):
        self.character_selection = character_selection

    def draw(self):
        return np.random.choice(self.character_selection)

    @property
    def n_param(self):
        return len(self.character_selection)

    @property
    def log_options(self):
        return np.log(len(self.character_selection))

    @classmethod
    def fit(cls, values):
        first_chars = set(v[0] if len(v) > 0 else "" for v in values)
        new_values = [v[1:] for v in values]
        return new_values, cls(list(first_chars))

    def __str__(self):
        return "["+"".join(self.character_selection)+"]"


class StringFreqDistribution(BaseDistribution):
    """String distribution that computes the frequency of characters.

    For the values supplied to the fit function, compute the string
    length distribution. Then for each character position compute the
    distribution of the characters.

    When drawing from the distribution, first draw a random string length
    from the distribution,
    and then for each position of the string, draw a character from the
    distribution for that position.

    It works particularly well with very strongly formatted strings, such as
    C25, A38, etc. For natural language, do not expect this to work
    particularly well.

    Parameters
    ----------
    str_lengths: array_like of int
        All string lengths available in the dataset.
    p_length: array_like of float
        Probability of each of the `str_lengths`. Has the same size.
    all_char_counts: list of tuple
        For each character position a tuple of length two should be supplied:
        * Available characters
        * Probability that those characters are selected.
    """

    aliases = ["char_freq"]

    def __init__(self, str_lengths, p_length, all_char_counts):
        self.str_lengths = str_lengths
        self.p_length = p_length
        self.all_char_counts = all_char_counts

    @classmethod
    def _fit(cls, values):
        values = values.astype(str)
        str_lengths, str_len_counts = np.unique([len(x) for x in values],
                                                return_counts=True)
        p_length = str_len_counts/len(values)
        all_char_counts = []
        for i_chr in range(np.max(str_lengths)):
            cur_chars, cur_char_counts = np.unique(
                [x[i_chr] for x in values if len(x) > i_chr],
                return_counts=True)
            all_char_counts.append((cur_chars,
                                    cur_char_counts/np.sum(cur_char_counts)))
        return cls(str_lengths, p_length, all_char_counts)

    def __str__(self):
        avg_len = np.sum([self.str_lengths[i]*self.p_length[i]
                          for i in range(len(self.str_lengths))])
        return f"String variable: avg # of characters: {avg_len} "

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": {
                "str_lengths": deepcopy(self.str_lengths),
                "p_length": deepcopy(self.p_length),
                "all_char_counts": deepcopy(self.all_char_counts),
            }
        }

    def draw(self):
        cur_str = ""
        str_len = np.random.choice(self.str_lengths, p=self.p_length)
        for i_chr in range(str_len):
            char_choices, p_choices = self.all_char_counts[i_chr]
            cur_str += np.random.choice(char_choices, p=p_choices)
        return cur_str
