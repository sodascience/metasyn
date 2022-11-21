"""Module containing string distributions."""

from __future__ import annotations

from abc import abstractmethod, ABC
import re
import random
import string
from typing import Iterable, Tuple, Sequence, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from metasynth.distribution.regex.optimizer import RegexOptimizer


class BaseRegexElement(ABC):
    """Base class for regex elements."""

    frac_used = 1.0

    def __init__(self, frac_used: float):
        if frac_used > 1 or frac_used < 0:
            raise ValueError(f"Error initializing RegexElement with fraction used {frac_used}."
                             f"Set the fraction to be used between 0 (never) and 1 (always).")
        self.frac_used = frac_used

    @abstractmethod
    def information_budget(self, regex_stat: Dict) -> float:
        """Addition to the AIC for the current regex."""

    @property
    def n_param(self) -> int:
        """int: The number of 'parameters' of the regex element."""
        return 2

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
    def from_string(cls, regex_str: str, frac_used: float=1.0
                    ) -> Optional[Tuple[BaseRegexElement, str]]:
        """Create a regex object from a regex string.

        Parameters
        ----------
        regex_str: str
            Regex string to process. Only single elements are supported.
        frac_used: float
            Fraction of the generated values for which the regex element is used.

        Returns
        -------
        Tuple[BaseRegexElement, str] or None:
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
        super().__init__(frac_used)
        self.min_digit = min_digit
        self.max_digit = max_digit
        self._prepare_regex()

    def _prepare_regex(self):
        regex_str = self.base_regex + r"{" + str(max(1, self.min_digit))
        regex_str += r"," + str(max(1, self.max_digit)) + r"}"
        self.match_regex = re.compile(regex_str)

    @property
    @abstractmethod
    def digit_options(self) -> int:
        """Return the number of options per digit."""
        return 0

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
        stats = optimizer.statistics
        new_regex = cls(stats["min_len"], stats["max_len"], frac_used=stats["frac_used"])
        delta_energy = RegexOptimizer.energy_from_values(values) - optimizer.energy
        info_budget = new_regex.information_budget(stats)
        return optimizer.new_values, delta_energy/info_budget, new_regex

    # def check_split(self, stats):
    #     digit_counts = stats["digits_used"]
    #     for digit_split in range(self.min_digit, self.max_digit):
    #         digits_used_left = np.copy(digit_counts)
    #         digits_used_left[digit_split] += np.sum(digit_counts[digit_split+1:])
    #         digits_used_left[digit_split+1:] = 0
    #         stat_left = {
    #             "n_values": stats["n_values"],
    #             "frac_used": self.frac_used,
    #             "digits_used": digits_used_left,
    #         }
    #         left_regex = self.__class__(self.min_digit, digit_split)
    #
    #         digits_used_right = np.copy(digit_counts)
    #         digits_used_right[0] = np.sum(digit_counts[:digit_split+1])
    #         digits_used_right[1:]
    #         right_regex = self.__class__(1, self.max_digit-digit_split+1)

    @classmethod
    def from_string(cls, regex_str, frac_used=1.0):
        match = re.search(cls.match_str, regex_str)
        if match is None:
            return None
        if all(x is None for x in match.groups()):
            return (cls(1, 1, frac_used=frac_used), regex_str[match.span()[1]:])
        return (cls(int(match.groups()[0]), int(match.groups()[1]), frac_used=frac_used),
                regex_str[match.span()[1]:])

    @classmethod
    def all_spans(cls, values: Sequence[str]) -> Sequence[Sequence[Tuple[int, int]]]:
        """Return match spans for a sequence of strings.

        Parameters
        ----------
        values:
            String values to match the regex to.

        Returns
        -------
        List[List[Tuple[int, int]]]:
            A list of spans for each of the values.
        """
        match_regex = re.compile(cls.base_regex+r"+")
        return [
            [match.span() for match in match_regex.finditer(val)]
            for val in values
        ]

    def information_budget(self, regex_stat: dict) -> float:
        if self.frac_used < 1e-8:
            return 2*self.n_param
        budget = 2*self.n_param
        n_values = regex_stat["n_values"]
        digit_range = self.max_digit-self.min_digit+1
        p_choose = regex_stat["frac_used"]/digit_range
        digits_factor = np.log(p_choose)
        for i_digit in range(self.min_digit, self.max_digit+1):
            i_count = regex_stat["digits_used"][i_digit]
            options_factor = -i_digit*np.log(self.digit_options)
            budget -= 2*i_count*(options_factor + digits_factor)
        num_unused = regex_stat["digits_used"][0]
        if num_unused > 0:
            budget -= 2*num_unused * np.log(num_unused/n_values)
        return budget


class DigitRegex(BaseRegexClass):
    """Regex element for repeating digits."""

    regex_str = r"^\d{1,}"
    match_str = r"^\\d(?:{(\d*),(\d*)})?"
    base_regex = r"\d"

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([str(np.random.randint(0, 10)) for _ in range(n_digit)])

    def __str__(self):
        return r"\d{"+str(self.min_digit)+","+str(self.max_digit)+"}"

    @property
    def digit_options(self) -> int:
        return 10


class AlphaNumericRegex(BaseRegexClass):
    """Regex element for repeating alphanumeric character."""

    regex_str = r"^\w{1,}"
    match_str = r"^\\w(?:{(\d*),(\d*)})?"
    base_regex = r"\w"

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters+string.digits) for _ in range(n_digit)])

    def __str__(self):
        return r"\w{"+str(self.min_digit)+","+str(self.max_digit)+"}"

    @property
    def digit_options(self) -> int:
        return len(string.ascii_letters+string.digits)


class LettersRegex(BaseRegexClass):
    """Regex element for repeating letters."""

    regex_str = r"^[a-zA-Z]{1,}"
    match_str = r"^\[a-zA-Z\](?:{(\d*),(\d*)})?"
    base_regex = r"[a-zA-Z]"

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters) for _ in range(n_digit)])

    def __str__(self):
        return "[a-zA-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"

    @property
    def digit_options(self) -> int:
        return len(string.ascii_letters)


class LowercaseRegex(BaseRegexClass):
    """Regex element for repeating lowercase letters."""

    regex_str = r"^[a-z]{1,}"
    match_str = r"^\[a-z\](?:{(\d*),(\d*)})?"
    base_regex = r"[a-z]"

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(n_digit)])

    def __str__(self):
        return "[a-z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"

    @property
    def digit_options(self) -> int:
        return len(string.ascii_lowercase)


class UppercaseRegex(BaseRegexClass):
    """Regex element for repeating uppercase letters."""

    regex_str = r"^[A-Z]{1,}"
    match_str = r"^\[A-Z\](?:{(\d*),(\d*)})?"
    base_regex = r"[A-Z]"

    def _draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_uppercase) for _ in range(n_digit)])

    def __str__(self):
        return "[A-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"

    @property
    def digit_options(self) -> int:
        return len(string.ascii_uppercase)


class AnyRegex(BaseRegexClass):
    """Regex that matches any character.

    For generation, it uses string.printable characters plus any additional
    ones that are manually set.

    Parameters
    ----------
    min_digit:
        Minimum number of digits for the regex.
    max_digit:
        Maximum number of digits for the regex.
    extra_char:
        Extra characters to draw from outside of string.printable.
    frac_used:
        Fraction of the values containing the regex.
    """

    def __init__(self, min_digit: int, max_digit: int,  # pylint: disable=super-init-not-called
                 extra_char: Optional[set[str]]=None,
                 frac_used: float=1.0):
        self.extra_char = set() if extra_char is None else extra_char
        super().__init__(min_digit, max_digit, frac_used)

    def _prepare_regex(self):
        self.all_char = string.printable + "".join(self.extra_char)
        self.all_char_set = set(self.all_char)

    @classmethod
    def fit_start(cls, values: Sequence[str]) -> Tuple[Iterable[str], BaseRegexElement]:
        new_values, _, regex = cls.fit(values)
        return new_values[1], regex

    @classmethod
    def all_spans(cls, values: Sequence[str]) -> Sequence[Sequence[Tuple[int, int]]]:
        return [[(0, len(v))] if len(v) > 0 else [] for v in values]

    @property
    def n_param(self) -> int:
        return 2 + len(self.extra_char)

    @property
    def digit_options(self) -> int:
        return len(string.printable) + len(self.extra_char)

    def _draw(self) -> str:
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(self.all_char) for _ in range(n_digit)])

    @classmethod
    def from_string(cls, regex_str, frac_used=1.0):
        match = re.search(r"^\.\[(.*)\](?:\{(\d+),(\d+)\})?", regex_str)
        if match is None:
            return None
        groups = match.groups()
        extra_char = set(groups[0])

        if all(x is None for x in groups[1:]):
            return cls(1, 1, extra_char), regex_str[match.span()[1]:]
        min_digit = int(groups[1])
        max_digit = int(groups[2])
        return (cls(min_digit, max_digit, extra_char, frac_used=frac_used),
                regex_str[match.span()[1]:])

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
        super().__init__(frac_used)
        self.character_selection = list(sorted(character_selection))

    def _draw(self) -> str:
        return np.random.choice(self.character_selection)

    @property
    def n_param(self) -> int:
        return 1 + len(self.character_selection)

    def information_budget(self, regex_stat) -> float:
        if self.frac_used < 1e-8:
            return 2*self.n_param
        n_options = len(self.character_selection)
        n_unused = regex_stat["digits_used"][0]
        n_used = regex_stat["digits_used"][1]

        budget_used = n_used*np.log(self.frac_used/n_options)
        if 1-self.frac_used > 1e-8:
            budget_unused = n_unused*np.log(1-self.frac_used)
        else:
            budget_unused = 0
        return 2*self.n_param - 2*(budget_used+budget_unused)

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
        return optimizer.new_values, gradient, cls(cur_characters,
                                                   optimizer.statistics["frac_used"])

    def __str__(self):
        return "["+"".join(self.character_selection)+"]"

    @classmethod
    def from_string(cls, regex_str, frac_used=1.0):
        match = re.search(r"^\[(.+)\]", regex_str)
        if match is None:
            return None
        return cls(list(match.groups()[0]), frac_used), regex_str[match.span()[1]:]


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
        stats = optimizer.statistics
        regex = SingleRegex(start_char + [character], stats["frac_used"])
        info_budget = regex.information_budget(stats)
        energy_grad = (start_energy - optimizer.energy)/info_budget
        if energy_grad > best_solution[1]:
            best_solution = (character, energy_grad, optimizer)
    return best_solution
