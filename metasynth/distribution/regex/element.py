"""Module containing string distributions."""

from __future__ import annotations

from abc import abstractmethod, ABC
from copy import deepcopy
import re
import random
import string
from typing import Iterable, Tuple, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from metasynth.distribution.regex.optimizer import RegexOptimizer


class BaseRegexElement(ABC):
    """Base class for regex elements."""

    frac_used = 1.0

    def information_criterion(self, n_char_removed: int) -> float:
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
        ic_add: float = 2*self.n_param + 2*self.log_options
        return ic_add/(n_char_removed+1e-8)

    @property
    def information_budget(self) -> float:
        """Addition to the AIC for the current regex."""
        return 2*self.n_param + 2*self.log_options

    @property
    def n_param(self) -> int:
        """int: The number of 'parameters' of the regex element"""
        return 2

    @property
    @abstractmethod
    def log_options(self) -> float:
        """float: The natural logarithm of the number of possibilities."""

    @classmethod
    @abstractmethod
    def fit(cls, values: Sequence[str]) -> Tuple[
            Tuple[Iterable[str], Iterable[str]], float, BaseRegexElement]:
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

    @classmethod
    @abstractmethod
    def fit_start(cls, values: Sequence[str]) -> Tuple[Iterable[str], BaseRegexElement]:
        """Match the regex to the values, sequential greedy algotithm from the left.

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
    def _draw(self) -> str:
        """Draw a random string from the regex element.

        Returns
        -------
        str:
            The string drawn from the regex distribution element.
        """

    def draw(self) -> str:
        """Draw a random string, considering how often the element is used.

        Returns
        -------
        random_string: String randomly drawn from the regex.
        """
        if np.random.rand() < self.frac_used:
            return self._draw()
        return ""

    @classmethod
    @abstractmethod
    def from_string(cls, regex_str):
        """Create a regex object from a regex string.

        Parameters
        ----------
        regex_str: str
            Regex string to process. Only single elements are supported.

        Returns
        -------
        BaseRegexElement or None:
            If the regex element is compatible with the string, then return the regex element
            that corresponds to the string. Otherwise return None.
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
    frac_used: Fraction of values that are matched at all.
    """
    regex_str = r""
    match_str = r""
    base_regex = r""

    def __init__(self, min_digit: int, max_digit: int, frac_used: float=1.0):
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.frac_used = frac_used
        regex_str = self.base_regex + r"{" + str(max(1, min_digit))
        regex_str += r"," + str(max(1, max_digit)) + r"}"
        self.match_regex = re.compile(regex_str)

    @classmethod
    def fit_start(cls, values: Sequence[str]) -> Tuple[Iterable[str], BaseRegexElement]:
        regex = re.compile(cls.regex_str)
        new_values = []

        min_match = 99999
        max_match = 0
        n_vals_matched = 0
        for val in values:
            match = regex.search(val)
            if match is None:
                n_match = 0
            else:
                n_match = match.span()[1]
                n_vals_matched += 1
                min_match = min(n_match, min_match)
                max_match = max(n_match, max_match)
            new_values.append(val[n_match:])
        if min_match == 99999:
            min_match = 0
            max_match = 1
        return new_values, cls(min_match, max_match, n_vals_matched/len(values))

    @classmethod
    def fit(cls, values: Sequence[str]) -> Tuple[
            Tuple[Iterable[str], Iterable[str]], float, BaseRegexElement]:
        spans = cls.all_spans(values)
        optimizer = RegexOptimizer(values, spans)
        optimizer.optimize()
        min_digit, max_digit, frac_used = optimizer.statistics
        new_regex = cls(min_digit, max_digit, frac_used)
        delta_energy = RegexOptimizer.energy_from_values(values) - optimizer.energy
        info_budget = new_regex.information_budget
        return optimizer.new_values, delta_energy/info_budget, new_regex

    @classmethod
    def from_string(cls, regex_str):
        match = re.search(cls.match_str, regex_str)
        if match is None:
            return None
        return cls(int(match.groups()[0]), int(match.groups()[1]))

    @classmethod
    def all_spans(cls, values: Sequence[str]) -> Sequence[Sequence[Tuple[int, int]]]:
        """Return match spans for a sequence of strings.

        Parameters
        ----------
        values: String values to match the regex to.

        Returns
        -------
        A list of spans for each of the values.
        """


class DigitRegex(BaseRegexClass):
    """Regex element for repeating digits."""
    regex_str = r"^\d{1,}"
    match_str = r"\\d{(\d*),(\d*)}"
    base_regex = r"\d"

    @property
    def log_options(self):
        return self.max_digit*np.log(10)

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([str(np.random.randint(0, 10)) for _ in range(n_digit)])

    def __str__(self):
        return r"\d{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class AlphaNumericRegex(BaseRegexClass):
    """Regex element for repeating alphanumeric character."""
    regex_str = r"^\w{1,}"
    match_str = r"\\w{(\d*),(\d*)}"
    base_regex = r"\w"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters+string.digits))

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters+string.digits) for _ in range(n_digit)])

    def __str__(self):
        return r"\w{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class LettersRegex(BaseRegexClass):
    """Regex element for repeating letters."""
    regex_str = r"^[a-zA-Z]{1,}"
    match_str = r"\[a-zA-Z\]{(\d*),(\d*)}"
    base_regex = r"[a-zA-Z]"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters))

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters) for _ in range(n_digit)])

    def __str__(self):
        return "[a-zA-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class LowercaseRegex(BaseRegexClass):
    """Regex element for repeating lowercase letters."""
    regex_str = r"^[a-z]{1,}"
    match_str = r"\[a-z\]{(\d*),(\d*)}"
    base_regex = r"[a-z]"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_lowercase))

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(n_digit)])

    def __str__(self):
        return "[a-z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class UppercaseRegex(BaseRegexClass):
    """Regex element for repeating uppercase letters."""
    regex_str = r"^[A-Z]{1,}"
    match_str = r"\[A-Z\]{(\d*),(\d*)}"
    base_regex = r"[A-Z]"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_uppercase))

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_uppercase) for _ in range(n_digit)])

    def __str__(self):
        return "[A-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class AnyRegex(BaseRegexElement):
    """Regex that matches any character.

    For generation, it uses string.printable characters plus any additional
    ones that are manually set.

    Parameters
    ----------
    min_digit: Minimum number of digits for the regex.
    max_digit: Maximum number of digits for the regex.
    extra_char: Extra characters to draw from outside of string.printable.
    frac_used: Fraction of the values containing the regex.
    """
    def __init__(self, min_digit: int, max_digit: int, extra_char: set=None, frac_used=1):
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.extra_char = set() if extra_char is None else extra_char
        self.frac_used = frac_used
        self.all_char = string.printable + "".join(self.extra_char)
        self.all_char_set = set(self.all_char)

    @classmethod
    def fit_start(cls, values: Sequence[str]) -> Tuple[Iterable[str], AnyRegex]:
        min_digit = np.min([len(v) for v in values])
        max_digit = np.max([len(v) for v in values])
        all_chars = "".join(values)
        extra_char = set(all_chars)-set(string.printable)
        new_values = ["" for _ in range(len(values))]
        frac_used = np.mean([len(values[i]) != len(new_values[i]) for i in range(len(values))])
        return new_values, cls(min_digit, max_digit, extra_char, frac_used)

    @classmethod
    def fit(cls, values: Sequence[str]) -> Tuple[
            Tuple[Iterable[str], Iterable[str]], float, BaseRegexElement]:
        new_values, new_regex = cls.fit_start(values)
        information_budget = new_regex.information_budget
        delta_energy = RegexOptimizer.energy_from_values(values)
        return (new_values, deepcopy(new_values)), delta_energy/information_budget, new_regex

    @property
    def n_param(self) -> int:
        return 1 + len(self.extra_char)

    @property
    def log_options(self) -> float:
        return self.max_digit*np.log(len(string.printable) + len(self.extra_char))

    def _draw(self) -> str:
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(self.all_char) for _ in range(n_digit)])

    @classmethod
    def from_string(cls, regex_str):
        match = re.search(r"\.\[(.*)\]\{(\d+),(\d+)\}", regex_str)
        if match is None:
            return None
        groups = match.groups()
        extra_char = set(groups[0])
        min_digit = int(groups[1])
        max_digit = int(groups[2])
        return cls(min_digit, max_digit, extra_char)

    def __str__(self):
        return f".[{''.join(self.extra_char)}]{{{self.min_digit},{self.max_digit}}}"


class SingleRegex(BaseRegexElement):
    """Regex element that is a choice on a set of (any) characters.

    Parameters
    ----------
    character_selection:
        The selection of characters available. An empty character ''
        is also allowed.
    """
    def __init__(self, character_selection, frac_used=1.0):
        self.character_selection = character_selection
        self.frac_used = frac_used

    def _draw(self) -> str:
        return np.random.choice(self.character_selection)

    @property
    def n_param(self) -> int:
        return 1 + len(self.character_selection)

    @property
    def log_options(self) -> float:
        return np.log(len(self.character_selection))

    @classmethod
    def fit_start(cls, values: Sequence[str]) -> Tuple[Iterable[str], SingleRegex]:
        first_chars = [v[0] for v in values if len(v) > 0]
        new_values = [v[1:] for v in values]
        return new_values, cls(list(set(first_chars)), len(first_chars)/len(values))

    @classmethod
    def fit(cls, values):
        vec = CountVectorizer(analyzer="char", lowercase=False)
        counts = vec.fit_transform(values).getnnz(0)
        count_dict = {c: counts[i] for c, i in vec.vocabulary_.items()}
        sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

        cur_characters = []
        best_solution = (None, -10.0)
        while True:
            new_solution = _find_best_gradient(values, sorted_counts, cur_characters)

            # Check to see if no character is left.
            if new_solution[0] is None:
                break

            new_char, new_grad, optimizer = new_solution
            if new_grad > best_solution[1]:
                best_solution = (optimizer, new_grad)
                cur_characters.append(new_char)
            else:
                break
        optimizer, gradient = best_solution
        return optimizer.new_values, gradient, cls(cur_characters, optimizer.statistics[2])

    def __str__(self):
        return "["+"".join(self.character_selection)+"]"

    @classmethod
    def from_string(cls, regex_str):
        match = re.search(r"\[(.+)\]", regex_str)
        if match is None:
            return None
        return cls(list(match.groups()[0]))


def _create_spans_char(values, *characters):
    """Create the spans for the values given a number of characters for the match."""
    char_list = "".join(re.escape(c) for c in characters)
    regex = re.compile(r"[" + char_list + r"]")
    return [
        [match.span() for match in regex.finditer(val)]
        for val in values
    ]


def _find_best_gradient(values, sorted_counts, start_char=None):
    """Find the best gradient by adding one additional character to the set."""
    def information_budget(n_char):
        n_param = n_char + 1
        return 2*n_param + 2*np.log(n_char)

    if start_char is None:
        start_char = []
    start_energy = RegexOptimizer.energy_from_values(values)
    best_solution = (None, -10.0, None)
    for character, _char_count in sorted_counts:
        if character in start_char:
            continue
        spans = _create_spans_char(values, character, *start_char)
        optimizer = RegexOptimizer(values, spans)
        optimizer.optimize()
        energy_grad = (start_energy - optimizer.energy)/information_budget(1+len(start_char))
        if energy_grad > best_solution[1]:
            best_solution = (character, energy_grad, optimizer)
    return best_solution
