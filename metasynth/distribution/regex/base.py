"""Module containing string distributions."""

from __future__ import annotations

from typing import Iterable, List, Union, Tuple, Type, Sequence, Set

import numpy as np

from metasynth.distribution.base import StringDistribution
from metasynth.distribution.regex.element import BaseRegexElement
from metasynth.distribution.regex.element import DigitRegex, AlphaNumericRegex
from metasynth.distribution.regex.element import LettersRegex, SingleRegex, AnyRegex
from metasynth.distribution.regex.element import UppercaseRegex, LowercaseRegex
from metasynth.distribution.regex.optimizer import RegexOptimizer


def _get_n_char_removed(values: Iterable[str], new_values: Iterable[str]) -> int:
    """Get the proportion of the characters that were resolved with the regex."""
    n_char = np.sum([len(val) for val in values])
    n_new_char = np.sum([len(val) for val in new_values])
    return n_char-n_new_char


class RegexDistribution(StringDistribution):
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

    aliases = ["RegexDistribution", "regex"]

    def __init__(self, re_list: Sequence[Union[BaseRegexElement, Tuple[str, float]]]):
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

    @classmethod
    def all_regex_classes(cls) -> List[Type[BaseRegexElement]]:
        """Return all regex element classes."""
        return [
            DigitRegex, AlphaNumericRegex, LowercaseRegex, UppercaseRegex,
            LettersRegex, SingleRegex, AnyRegex,
        ]

    @classmethod
    def _fit(cls, values):
        if np.sum([len(val) for val in values]) == 0:
            return cls([])

        best_solution = None
        regex_classes = cls.all_regex_classes()

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, gradient, regexer = re_class.fit(list(values))
            print(re_class, gradient, regexer)
            print("\n\n\n")
            if best_solution is None or gradient > best_solution["gradient"]:
                best_solution = {
                    "re": regexer,
                    "gradient": gradient,
                    "values": new_values,
                }
        if best_solution is None:
            return cls([])
        left_fit = cls._fit(best_solution["values"][0])
        middle_fit = cls([best_solution["re"]])
        right_fit = cls._fit(best_solution["values"][1])
        return left_fit + middle_fit + right_fit

    @classmethod
    def _fit_start(cls, values):
        n_char = np.sum([len(val) for val in values])
        if n_char == 0:
            return cls([])
        best_solution = None
        regex_classes = cls.all_regex_classes()

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, regexer = re_class.fit(list(values))
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
            "name": self.name,
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
    aliases = ["UniqueRegexDistribution", "regex_unique"]

    def __init__(self, re_list: Sequence[Union[BaseRegexElement, Tuple[str, float]]]):
        super().__init__(re_list)
        self.key_set: Set[str] = set()

    # @property
    # def n_options(self) -> float:
    #     """float: approximate number of options for the regex."""
    #     cur_log_options = 0.0
    #     for rex in self.re_list:
    #         cur_log_options += rex.log_options
    #     if cur_log_options > 30:
    #         return np.inf
    #     return np.exp(cur_log_options)

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> str:
        # n_options = self.n_options
        # if not np.isinf(n_options):
            # if len(self.key_set)/n_options >= 0.99:
                # raise ValueError("Found 99% of the possible values, running out of possibilities.")
        n_try = 0
        while n_try < 1e5:
            new_val = super().draw()
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val
            n_try +=1
        raise ValueError("Failed to draw unique string after 100.000 tries.")

    def information_criterion(self, values):
        return 99999
        # The information criterion is relative to the non-unique version.
        # values = values.dropna()
        # if len(set(values)) != len(values):
        #     return 999*len(values)
        #
        # n_options = self.n_options
        # if np.isinf(n_options):
        #     return 1
        #
        # return 1-2*np.sum(np.log(n_options/np.arange(n_options, n_options-len(values), -1)))
