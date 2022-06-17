"""Module containing string distributions."""

from __future__ import annotations

from abc import abstractmethod, ABC
import re
import random
import string
from typing import Iterable, List, Union, Tuple, Type, Sequence, Set

import numpy as np

from metasynth.distribution.base import BaseDistribution


def _get_n_char_removed(values: Iterable, new_values: Iterable):
    """Get the proportion of the characters that were resolved with the regex."""
    n_char = np.sum([len(val) for val in values])
    n_new_char = np.sum([len(val) for val in new_values])
    return (n_char-n_new_char)


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

    def __init__(self, re_list: Sequence[Union[BaseRegexElement, Tuple[str, float]]]):
        # if len(re_list) > 0 and not isinstance(re_list[0], BaseRegexElement):

        self.re_list = []
        for re_elem in re_list:
            if isinstance(re_elem, BaseRegexElement):
                self.re_list.append(re_elem)
                continue
            regex_str, frac_used = re_elem
            regex_dist_elem = None
            for regex_class in self.all_regex_classes():
                regex_dist_elem = regex_class.from_string(regex_str)
                if regex_dist_elem is not None:
                    break
            if regex_dist_elem is None:
                raise ValueError(f"Unrecognized regex element '{regex_str}'")
            regex_dist_elem.frac_used = frac_used
            self.re_list.append(regex_dist_elem)
        # else:
            # self.re_list = re_list

    @classmethod
    def all_regex_classes(cls) -> List[Type[BaseRegexElement]]:
        """Return all regex element classes."""
        return [
            DigitRegex, AlphaNumericRegex, LowercaseRegex, UppercaseRegex,
            LettersRegex, SingleRegex, AnyRegex,
        ]

    @classmethod
    def _fit(cls, values):
        n_char = np.sum([len(val) for val in values])
        if n_char == 0:
            return cls([])
        best_solution = None
        regex_classes = cls.all_regex_classes()

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, regexer = re_class.fit([v for v in values])
            n_char_removed = _get_n_char_removed(values, new_values)
            cur_ic = regexer.information_criterion(n_char_removed)
            # if cur_ic > 1.5:
            # continue
            if best_solution is None or cur_ic < best_solution["ic"]:
                best_solution = {
                    "ic": cur_ic,
                    "re": regexer,
                    "values": new_values,
                }
        if best_solution is None:
            return cls([])
        return cls([best_solution["re"]]) + cls._fit(best_solution["values"])

    def __add__(self, other: RegexDistribution) -> RegexDistribution:
        return self.__class__(self.re_list + other.re_list)

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
                    "re_list": [(str(x), x.frac_used) for x in self.re_list],
                }
        }


class UniqueRegexDistribution(RegexDistribution):
    """Unique variant of the regex distribution.

    Same as the normal regex distribution, but checks whether a key
    has already been used.

    Parameters
    ----------
    re_list: list of BaseRegexElement
        List of basic regex elements in the order that they occur.
    """

    is_unique = True
    aliases = ["regex_unique"]

    def __init__(self, re_list: Sequence[Union[BaseRegexElement, Tuple[str, float]]]):
        super().__init__(re_list)
        self.key_set: Set[str] = set()

    @property
    def n_options(self) -> float:
        """float: approximate number of options for the regex."""
        cur_log_options = 0.0
        for rex in self.re_list:
            print(rex.log_options)
            cur_log_options += rex.log_options
        if cur_log_options > 30:
            return np.inf
        return np.exp(cur_log_options)

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> str:
        n_options = self.n_options
        if not np.isinf(n_options):
            if len(self.key_set)/n_options >= 0.99:
                raise ValueError("Found 99% of the possible values, running out of possibilities.")
        while True:
            new_val = super().draw()
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val

    def information_criterion(self, values):
        # The information criterion is relative to the non-unique version.
        values = values.dropna()
        if len(set(values)) != len(values):
            return 999*len(values)

        n_options = self.n_options
        if np.isinf(n_options):
            return 1

        return 1-2*np.sum(np.log(n_options/np.arange(n_options, n_options-len(values), -1)))


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
        ic_add = 2*self.n_param + 2*self.log_options
        return ic_add/(n_char_removed+1e-8)

    @property
    def n_param(self) -> int:
        """int: The number of 'parameters' of the regex element"""
        return 1

    @property
    @abstractmethod
    def log_options(self) -> float:
        """float: The natural logarithm of the number of possibilities."""

    @classmethod
    @abstractmethod
    def fit(cls, values: Sequence[str]) -> Tuple[Iterable[str], BaseRegexElement]:
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
    def _draw(self) -> str:
        """Draw a random string from the regex element.

        Returns
        -------
        str:
            The string drawn from the regex distribution element.
        """

    def draw(self) -> str:
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

    @abstractmethod
    def matches(self, values: Sequence[str]) -> Iterable[bool]:
        pass


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
    regex_str = r""
    match_str = r""
    base_regex = r""

    def __init__(self, min_digit: int, max_digit: int, frac_used: float=1.0):
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.frac_used = frac_used
        regex_str = self.base_regex + r"{" + str(max(1, min_digit)) + r"," + str(max(1, max_digit)) + r"}"
        self.match_regex = re.compile(regex_str)

    @classmethod
    def fit(cls, values: Sequence[str]):
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
    def fit_middle(cls, values: Sequence[str]):
        pass

    @classmethod
    def from_string(cls, regex_str):
        match = re.search(cls.match_str, regex_str)
        if match is None:
            return None
        return cls(int(match.groups()[0]), int(match.groups()[1]))

    def matches(self, value):
        return [match.span() for match in self.match_regex.finditer(value)]

    @classmethod
    def all_spans(cls, values: Sequence[str]):
        match_regex = re.compile(cls.base_regex+r"+")
        return [
            [match.span() for match in match_regex.finditer(val)]
            for val in values
        ]


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
    def __init__(self, min_digit: int, max_digit: int, extra_char: set=None, frac_used=1):
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.extra_char = set() if extra_char is None else extra_char
        self.frac_used = frac_used
        self.all_char = string.printable + "".join(self.extra_char)
        self.all_char_set = set(self.all_char)

    @classmethod
    def fit(cls, values: Sequence[str]) -> Tuple[Iterable[str], AnyRegex]:
        min_digit = np.min([len(v) for v in values])
        max_digit = np.max([len(v) for v in values])
        all_chars = "".join(values)
        extra_char = set(all_chars)-set(string.printable)
        new_values = ["" for _ in range(len(values))]
        frac_used = np.mean([len(values[i]) != len(new_values[i]) for i in range(len(values))])
        return new_values, cls(min_digit, max_digit, extra_char, frac_used)

    @property
    def n_param(self) -> int:
        return 1 + len(self.extra_char)

    @property
    def log_options(self) -> float:
        return self.max_digit*np.log(len(string.printable) + len(self.extra_char))

    def _draw(self):
        return random.choice(self.all_char)

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

    def matches(self, values: Sequence[str]) -> Iterable[bool]:
        return [len(set(v)-self.all_char_set) == 0 for v in values]


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

    def _draw(self):
        return np.random.choice(self.character_selection)

    @property
    def n_param(self):
        return len(self.character_selection)

    @property
    def log_options(self):
        return np.log(len(self.character_selection))

    @classmethod
    def fit(cls, values):
        first_chars = [v[0] for v in values if len(v) > 0]
        new_values = [v[1:] for v in values]
        return new_values, cls(list(set(first_chars)), len(first_chars)/len(values))

    def __str__(self):
        return "["+"".join(self.character_selection)+"]"

    @classmethod
    def from_string(cls, regex_str):
        match = re.search(r"\[(.+)\]", regex_str)
        if match is None:
            return None
        return cls(list(match.groups()[0]))

    def matches(self, value):
        return [(i, i+1) for i, val_chr in enumerate(value) if val_chr in self.character_selection]
