"""Module with utilities that supports the distributions."""

import inspect
import importlib
from typing import Sequence, Tuple

import numpy as np

from metasynth.distribution.base import BaseDistribution

SIDE_LEFT = -1
SIDE_RIGHT = -2


def get_dist_class(name):
    """Obtain a distribution and fit arguments from a name

    For example, if we use "faker.city.nl_NL", we should get the FakerDistribution
    with the key word arguments/dictionary: {faker_type: city, locale: nl_NL}.

    Parameters
    ----------
    name: str
        Name of the distribution.

    Returns
    -------
    tuple of BaseDistribution, dict:
        The class that relates to the distribution name and the fit
        keyword arguments as a dictionary. Most distributions will have
        an empty dictionary.
    """
    modules = [
        "metasynth.distribution.continuous",
        "metasynth.distribution.discrete",
        "metasynth.distribution.faker",
        "metasynth.distribution.regex.base",
        "metasynth.distribution.categorical"
    ]

    # Iterate over all distribution modules
    for module_str in modules:
        module = importlib.import_module(module_str)
        for _, dist_class in inspect.getmembers(module, inspect.isclass):
            # Check if it comes originally from the current module.
            if not dist_class.__module__ == module.__name__:
                continue

            # Check if it is a distribution
            if not issubclass(dist_class, BaseDistribution):
                continue

            # Ask the distribution if the name belongs to them
            if dist_class.is_named(name):
                return dist_class, dist_class.fit_kwargs(name)
    raise ValueError(f"Cannot find distribution with name {name}")


class RegexOptimizer():
    def __init__(self, values: Sequence[str], spans: Sequence[Sequence[Tuple[int, int]]]):
        self.values = values
        self.spans = spans
        self.cur_solution = np.zeros(len(values), dtype=int)
        self.max_len = np.max([len(v) for v in values])
        self.length_dist = np.zeros(2*self.max_len, dtype=int)
        for i_val in range(len(values)):
            cur_span_choice = spans[i_val]
            cur_val = values[i_val]
            if len(cur_span_choice) > 0:
                i_choice: int = int(np.argmax([x[1]-x[0] for x in cur_span_choice]))
                cur_span = cur_span_choice[i_choice]
                self.cur_solution[i_val] = i_choice
                self.length_dist[-(cur_span[0])] += 1
                self.length_dist[len(cur_val)-cur_span[1]] += 1
            else:
                self.cur_solution[i_val] = np.random.choice([SIDE_LEFT, SIDE_RIGHT])
                self.length_dist[0] += 1
                if self.cur_solution[i_val] == SIDE_RIGHT:
                    self.length_dist[len(cur_val)] += 1
                else:
                    self.length_dist[-len(cur_val)] += 1
        self.check_integrity()

    @property
    def energy(self) -> float:
        assert np.all(self.length_dist >= 0)
        left_energy = np.arange(0, self.max_len)*np.log(self.length_dist[:self.max_len]+1)
        right_energy = np.arange(self.max_len, 0, -1)*np.log(self.length_dist[-self.max_len:]+1)
        return np.sum(left_energy + right_energy)

    def energy_rm(self, length: int) -> float:
        n_i = self.length_dist[length]
        return np.abs(length)*np.log(n_i/(n_i+1))

    def energy_add(self, length: int) -> float:
        n_i = self.length_dist[length]
        return np.abs(length)*np.log((n_i+2)/(n_i+1))

    def optimize(self) -> None:
        while self._optimize_round():
            # print(self.energy)
            pass

    def _optimize_round(self) -> bool:
        has_changed: bool = False
        for i_val, cur_val in enumerate(self.values):
            cur_span_choice = self.spans[i_val]
            if len(cur_span_choice) == 0:
                chosen_side = self.cur_solution[i_val]
                diff = self.length_dist[len(cur_val)] - self.length_dist[-len(cur_val)]
                if chosen_side == SIDE_LEFT and diff > 1:
                    self.length_dist[-len(cur_val)] -= 1
                    self.length_dist[len(cur_val)] += 1
                    self.cur_solution[i_val] = SIDE_RIGHT
                    # print(i_val, "Swap left to right")
                    has_changed = True
                elif chosen_side == SIDE_RIGHT and diff < -1:
                    self.length_dist[len(cur_val)] -= 1
                    self.length_dist[-len(cur_val)] += 1
                    self.cur_solution[i_val] = SIDE_LEFT
                    # print(i_val, "Swap right to left")

                    has_changed = True
                continue
            chosen_span = self.cur_solution[i_val]
            i_start, i_temp = cur_span_choice[chosen_span]
            i_end = len(cur_val)-i_temp
            best_choice = (chosen_span, 0.0)
            for j_choice in range(len(cur_span_choice)):
                if j_choice == self.cur_solution[i_val]:
                    continue
                j_start, j_temp = cur_span_choice[j_choice]
                j_end = len(cur_val)-j_temp
                delta_E = (self.energy_rm(-i_start) + self.energy_rm(i_end) +
                           self.energy_add(-j_start) + self.energy_add(j_end))
                if delta_E - best_choice[1] > -1e-8:
                    continue
                best_choice = (j_choice,  delta_E)
            if best_choice[1] < 0:
                has_changed = True
                # print(i_val, chosen_span, best_choice[0], self.spans[i_val])
                self.set_span(i_val, best_choice[0])
        return has_changed

    def set_span(self, i_val, i_span) -> None:
        old_span = self.spans[i_val][self.cur_solution[i_val]]
        new_span = self.spans[i_val][i_span]
        val_len = len(self.values[i_val])

        # First remove old ones
        self.length_dist[-old_span[0]] -= 1
        self.length_dist[val_len-old_span[1]] -= 1

        # Add new ones
        self.length_dist[-new_span[0]] += 1
        self.length_dist[val_len-new_span[1]] += 1
        self.cur_solution[i_val] = i_span

    def check_integrity(self):
        if np.sum(self.length_dist) != 2*len(self.cur_solution):
            print(f"{np.sum(self.length_dist)}, {len(self.cur_solution != -1)}")
            assert False

    @property
    def new_values(self):
        left_values = []
        right_values = []
        for i_val, val in enumerate(self.values):
            if self.cur_solution[i_val] == SIDE_LEFT:
                left_values.append(val)
                right_values.append("")
            elif self.cur_solution[i_val] == SIDE_RIGHT:
                left_values.append("")
                right_values.append(val)
            else:
                span = self.spans[i_val][self.cur_solution[i_val]]
                left_values.append(val[:span[0]])
                right_values.append(val[span[1]:])
        return left_values, right_values

    @property
    def statistics(self):
        match_lengths = []
        for i_val in range(len(self.values)):
            if len(self.spans[i_val]) > 0:
                span = self.spans[i_val][self.cur_solution[i_val]]
                match_lengths.append(span[1]-span[0])

        if len(match_lengths) == 0:
            return 1, 1, 0.0

        min_len = np.min(match_lengths)
        max_len = np.max(match_lengths)
        frac_used = len(match_lengths)/len(self.values)
        return min_len, max_len, frac_used

    @staticmethod
    def energy_from_values(values: Sequence[str]) -> float:
        length_array = np.array([len(v) for v in values])
        lengths, counts = np.unique(length_array, return_counts=True)
        return np.sum(lengths*np.log(counts+1))
