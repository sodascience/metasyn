"""Module containing string distributions."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple, Type, Union

import numpy as np

from metasyn.distribution.base import metadist, BaseDistribution
from metasyn.distribution.legacy.regex.element import (AlphaNumericRegex, AnyRegex,
                                                       BaseRegexElement, DigitRegex,
                                                       LettersRegex, LowercaseRegex,
                                                       SingleRegex, UppercaseRegex)
from metasyn.distribution.legacy.regex.optimizer import RegexOptimizer


def _get_gradient_start(values: Sequence[str], new_values: Sequence[str],
                        regex_elem: BaseRegexElement) -> float:
    """Get the proportion of the characters that were resolved with the regex."""
    digits_used: List[int] = []
    for i_val, old_val in enumerate(values):
        new_val = new_values[i_val]
        d_len = len(old_val) - len(new_val)
        if len(digits_used) <= d_len:
            digits_used += (d_len - len(digits_used) + 1)*[0]
        digits_used[d_len] += 1
    regex_stat = {
        "n_values": len(values),
        "frac_used": regex_elem.frac_used,
        "digits_used": digits_used,
    }
    old_energy = RegexOptimizer.energy_from_values(values)
    new_energy = RegexOptimizer.energy_from_values(new_values)
    energy_budget = regex_elem.information_budget(regex_stat)

    delta_energy = old_energy - new_energy
    return delta_energy/energy_budget


@metadist(implements="core.regex", var_type="string")
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

    implements = "core.regex"

    def __init__(self, re_list: Union[Sequence[Union[BaseRegexElement, Tuple[str, float]]], str]):
        self.re_list = []
        if isinstance(re_list, str):
            self.re_list = self._unpack_regex(re_list)
            return

        for re_elem in re_list:
            if isinstance(re_elem, BaseRegexElement):
                self.re_list.append(re_elem)
                continue
            regex_str, frac_used = re_elem
            regex_return = None
            for regex_class in self.all_regex_classes():
                regex_return = regex_class.from_string(regex_str, frac_used)
                if regex_return is not None and regex_return[1] == "":
                    break
            if regex_return is None:
                raise ValueError(f"Unrecognized regex element '{regex_str}'")
            regex_dist_elem = regex_return[0]
            self.re_list.append(regex_dist_elem)

    @classmethod
    def all_regex_classes(cls) -> List[Type[BaseRegexElement]]:
        """Return all regex element classes."""
        return [
            DigitRegex, AlphaNumericRegex, LowercaseRegex, UppercaseRegex,
            LettersRegex, SingleRegex, AnyRegex,
        ]

    def _unpack_regex(self, regex_str: str):
        if len(regex_str) == 0:
            return []
        for re_elem_class in self.all_regex_classes():
            ret = re_elem_class.from_string(regex_str)
            if ret is not None:
                return [ret[0]] + self._unpack_regex(ret[1])
        raise ValueError("Failed to determine regex from '" + regex_str + "'")

    @classmethod
    def _fit(cls, values, mode: str = "fast"):
        if mode == "fast":
            return cls._fit_fast(values)
        return cls._fit_slow(values)

    @classmethod
    def _fit_slow(cls, values):
        if np.sum([len(val) for val in values]) == 0:
            return cls([])

        best_solution = None
        regex_classes = cls.all_regex_classes()

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, gradient, regexer = re_class.fit(list(values))
            if best_solution is None or gradient > best_solution["gradient"]:
                best_solution = {
                    "re": regexer,
                    "gradient": gradient,
                    "values": new_values,
                }
        if best_solution is None:
            return cls([])
        left_fit = cls._fit_slow(best_solution["values"][0])
        middle_fit = cls([best_solution["re"]])
        right_fit = cls._fit_slow(best_solution["values"][1])
        return left_fit + middle_fit + right_fit

    @classmethod
    def _fit_fast(cls, values):
        n_char = np.sum([len(val) for val in values])
        if n_char == 0:
            return cls([])
        best_solution = None
        regex_classes = cls.all_regex_classes()

        # Iterate over all RegexElements and find the best one.
        for re_class in regex_classes:
            new_values, regexer = re_class.fit_start(list(values))
            # n_char_removed = _get_n_char_removed(values, new_values)
            # cur_ic = regexer.information_criterion(n_char_removed)
            # if cur_ic > 1.5:
            # continue
            gradient = _get_gradient_start(values, new_values, regexer)
            if best_solution is None or gradient > best_solution["gradient"]:
                best_solution = {
                    "gradient": gradient,
                    "re": regexer,
                    "values": new_values,
                }
        if best_solution is None:
            return cls([])
        return cls([best_solution["re"]]) + cls._fit_fast(best_solution["values"])

    def __add__(self, other: RegexDistribution) -> RegexDistribution:
        return self.__class__(self.re_list + other.re_list)

    def draw(self):
        cur_str = ""
        for rex in self.re_list:
            cur_str += rex.draw()
        return cur_str

    @property
    def regex_string(self) -> str:
        """String representation of the string."""
        return "".join([str(x) for x in self.re_list])

    def _param_dict(self):
        return {
            "re_list": [(str(x), x.frac_used) for x in self.re_list],
        }

    @classmethod
    def _param_schema(cls):
        return {
            "re_list": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "number"}],
                    "minItems": 2,
                    "additionalItems": False
                }
            }
        }

    @classmethod
    def default_distribution(cls):
        return cls([(r"\d{3,4}", 0.67)])


@metadist(implements="core.unique_regex", var_type="string", is_unique=True)
class UniqueRegexDistribution(RegexDistribution):
    """Unique variant of the regex distribution.

    Same as the normal regex distribution, but checks whether a key
    has already been used.

    Parameters
    ----------
    re_list: list of BaseRegexElement
        List of basic regex elements in the order that they occur.
    """

    def __init__(self, re_list: Sequence[Union[BaseRegexElement, Tuple[str, float]]]):
        super().__init__(re_list)
        self.key_set: Set[str] = set()

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> str:
        n_try = 0
        while n_try < 1e5:
            new_val = super().draw()
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val
            n_try += 1
        raise ValueError("Failed to draw unique string after 100.000 tries.")

    def information_criterion(self, values):
        return 99999
