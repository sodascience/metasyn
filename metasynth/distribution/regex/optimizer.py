"""Module containing the optimizer class for the regex distribution."""

from typing import Sequence, Tuple, List, Dict, Any
import numpy as np
import numpy.typing as npt

# Used for side assignment of values with no regex match
SIDE_LEFT = -1
SIDE_RIGHT = -2


class RegexOptimizer():
    r"""Optimizer class for fitting regexes to lists of strings.

    As input it takes the list of values and a list of matching spans
    for each of these values. With that it tries to find the optimal span assignment.
    It does so in a greedy way, where it optimizes each value step by step.

    As a result of the assignment, the values are split: one part left of the regex and
    another part right of the regex. The goal is to optimize the parts on the left and right
    to ensure that is as close to optimal as possible.

    The objective value of the optimization for this is defined for a list of values as follows:
    take the cumulative distribution of the lengths of the values > 0 and compute:
    \sum_i=1 \log (C_i + 1).
    , where C_i is the number of values with length i or more.

    Parameters
    ----------
    values:
        Values to which the regex is being applied.
    spans:
        Positions/spans where the regex can be applied.
    """

    def __init__(self, values: Sequence[str], spans: Sequence[Sequence[Tuple[int, int]]]):
        self.values = values
        self.spans = spans
        self.cur_solution = np.zeros(len(values), dtype=int)
        self.max_len = np.max([len(v) for v in values])
        self.left_cum_dist = np.zeros(self.max_len, dtype=int)
        self.right_cum_dist = np.zeros(self.max_len, dtype=int)
        for i_val, cur_val in enumerate(values):
            cur_span_choice = spans[i_val]
            if len(cur_span_choice) > 0:
                i_choice: int = int(np.argmax([x[1]-x[0] for x in cur_span_choice]))
                cur_span = cur_span_choice[i_choice]
                self.cur_solution[i_val] = i_choice
                self.left_cum_dist[:cur_span[0]] += 1
                self.right_cum_dist[:len(cur_val)-cur_span[1]] += 1
            else:
                self.cur_solution[i_val] = np.random.choice([SIDE_LEFT, SIDE_RIGHT])
                if self.cur_solution[i_val] == SIDE_LEFT:
                    self.left_cum_dist[:len(cur_val)] += 1
                else:
                    self.right_cum_dist[:len(cur_val)] += 1

    def optimize(self) -> None:
        """Optimize the positions until no improvement can be found."""
        while self.optimize_round():
            pass

    def optimize_round(self) -> bool:
        """Do one round of optimization.

        Returns
        -------
        has_changed: Whether the optimization round has changed any assignment.
        """
        has_changed: bool = False

        # Iterate over all values
        for i_val, cur_val in enumerate(self.values):
            cur_span_choice = self.spans[i_val]
            if len(cur_span_choice) == 0:
                chosen_side = self.cur_solution[i_val]

                # If assigned to the left, attempt right assignment.
                if chosen_side == SIDE_LEFT:
                    delta_energy = self.energy_move(self.left_cum_dist, len(cur_val), 0)
                    delta_energy += self.energy_move(self.right_cum_dist, 0, len(cur_val))
                    if delta_energy < -1e-8:
                        self.left_cum_dist[:len(cur_val)] -= 1
                        self.right_cum_dist[:len(cur_val)] += 1
                        self.cur_solution[i_val] = SIDE_RIGHT
                        has_changed = True

                # If assigned to the right, attempt left assignment
                elif chosen_side == SIDE_RIGHT:
                    delta_energy = self.energy_move(self.right_cum_dist, len(cur_val), 0)
                    delta_energy += self.energy_move(self.left_cum_dist, 0, len(cur_val))
                    if delta_energy < -1e-8:
                        self.left_cum_dist[:len(cur_val)] += 1
                        self.right_cum_dist[:len(cur_val)] -= 1
                        self.cur_solution[i_val] = SIDE_LEFT
                        has_changed = True
                continue

            # Only reached if there is at least one match.
            chosen_span = self.cur_solution[i_val]
            i_start, i_temp = cur_span_choice[chosen_span]
            i_end = len(cur_val)-i_temp
            best_choice = (chosen_span, 0.0)

            # Try all spans except the one we already have and find the one with the lowest energy.
            for j_choice, j_span in enumerate(cur_span_choice):
                if j_choice == self.cur_solution[i_val]:
                    continue
                j_start, j_temp = j_span
                j_end = len(cur_val)-j_temp
                delta_energy = (self.energy_move(self.left_cum_dist, i_start, j_start) +
                                self.energy_move(self.right_cum_dist, i_end, j_end))
                if delta_energy - best_choice[1] > -1e-8:
                    continue
                best_choice = (j_choice,  delta_energy)
            if best_choice[1] < 0:
                has_changed = True
                self.set_span(i_val, best_choice[0])
        return has_changed

    @property
    def energy(self) -> float:
        """Compute the energy of the current assignment/solution."""
        left_energy = np.sum(np.log(self.left_cum_dist+1))
        right_energy = np.sum(np.log(self.right_cum_dist+1))
        return left_energy + right_energy

    def energy_move(self, dist: npt.NDArray[np.int_], len_src: int, len_dst: int) -> float:
        """Compute the energy to change the length on one side.

        Parameters
        ----------
        dist:
            Cumulative distribution of the current lengths.
            (self.left_cum_dist or self.right_cum_dist)
        len_src:
            Current length on this side.
        len_dst:
            New length on the same side.

        Returns
        -------
        delta_energy:
            Difference in energy [after-before].
        """
        if len_dst > len_src:
            return np.sum(np.log((dist[len_src:len_dst]+2) /
                                 (dist[len_src:len_dst]+1)))
        if len_dst < len_src:
            assert np.all(dist[len_dst:len_src] > 0), (dist[len_dst:len_src], len_src, len_dst)
            return np.sum(np.log((dist[len_dst:len_src] /
                                  (dist[len_dst:len_src]+1))))
        return 0

    def _check_integrity(self):
        """Check the integrity of the assignments."""
        n_sum_left = 0
        n_sum_right = 0
        for i_val, cur_val in enumerate(self.values):
            i_span = self.cur_solution[i_val]
            if i_span == SIDE_LEFT:
                n_sum_left += len(cur_val)
            elif i_span == SIDE_RIGHT:
                n_sum_right += len(cur_val)
            else:
                n_sum_left += self.spans[i_val][i_span][0]
                n_sum_right += len(cur_val) - self.spans[i_val][i_span][1]
        expected_sum_left = np.sum(self.left_cum_dist)
        expected_sum_right = np.sum(self.right_cum_dist)
        if n_sum_left != expected_sum_left or n_sum_right != expected_sum_right:
            print(n_sum_left, expected_sum_left, n_sum_right, expected_sum_right)
            print(self.left_cum_dist)
            print(self.right_cum_dist)
            print(self.values)
            print(self.spans)
            print(self.cur_solution)
            assert False

    @staticmethod
    def energy_from_values(values: Sequence[str]) -> float:
        """Compute the energy of a sequence of strings.

        Independent of the current optimized assignment.

        Parameters
        ----------
        values:
            Values to compute the energy of.

        Returns
        -------
        float:
            Computed energy.
        """
        length_array = np.array([len(v) for v in values])
        lengths, counts = np.unique(length_array, return_counts=True)
        cum_len = np.zeros(np.max(lengths))

        # Compute the cumulative lengths.
        for i_len, cur_len in enumerate(lengths):
            cur_counts = counts[i_len]
            cum_len[:cur_len] += cur_counts
        return np.sum(np.log(cum_len + 1))

    def set_span(self, i_val: int, i_span: int) -> None:
        """Change the assignment for a value.

        Parameters
        ----------
        i_val:
            Index of the value to be reassigned.
        i_span:
            Index of the span for that value to be assigned to.
        """
        old_span = self.spans[i_val][self.cur_solution[i_val]]
        new_span = self.spans[i_val][i_span]

        # First change the ones on the left
        if old_span[0] > new_span[0]:
            self.left_cum_dist[new_span[0]:old_span[0]] -= 1
        else:
            self.left_cum_dist[old_span[0]:new_span[0]] += 1

        # Then on the right.
        old_right, new_right = len(self.values[i_val]) - np.array([old_span[1], new_span[1]])
        if old_right > new_right:
            self.right_cum_dist[new_right:old_right] -= 1
        else:
            self.right_cum_dist[old_right:new_right] += 1

        # Update the current solution.
        self.cur_solution[i_val] = i_span

    @property
    def new_values(self) -> Tuple[List[str], List[str]]:
        """Get the values on the left and right with the current assignment."""
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
    def statistics(self) -> Dict[str, Any]:
        """Get the minimum/maximum length of the substituted regex and fraction of assignments."""
        # Compute lengths for which the regex substitutes.
        match_lengths: List[int] = []
        for i_val in range(len(self.values)):
            if len(self.spans[i_val]) > 0:
                span = self.spans[i_val][self.cur_solution[i_val]]
                match_lengths.append(span[1]-span[0])

        if len(match_lengths) == 0:
            return {
                "min_len": 1,
                "max_len": 1,
                "frac_used": 0,
                "digits_used": np.zeros(2, dtype=int),
                "n_values": len(self.values),
            }

        min_len = np.min(match_lengths)
        max_len = np.max(match_lengths)
        frac_used = len(match_lengths)/len(self.values)
        digits_used = np.zeros(max_len+1, dtype=int)
        length, counts = np.unique(match_lengths, return_counts=True)
        for i_len, cur_len in enumerate(length):
            digits_used[cur_len] = counts[i_len]
        digits_used[0] = len(self.values)-len(match_lengths)
        # digits_used = digits_used/np.sum(digits_used)
        return {
            "min_len": min_len,
            "max_len": max_len,
            "frac_used": frac_used,
            "digits_used": digits_used,
            "n_values": len(self.values),
        }
