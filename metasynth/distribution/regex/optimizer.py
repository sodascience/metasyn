from typing import Sequence, Tuple
import numpy as np

SIDE_LEFT = -1
SIDE_RIGHT = -2


class RegexOptimizer():
    def __init__(self, values: Sequence[str], spans: Sequence[Sequence[Tuple[int, int]]]):
        self.values = values
        self.spans = spans
        self.cur_solution = np.zeros(len(values), dtype=int)
        self.max_len = np.max([len(v) for v in values])
        self.left_cum_dist = np.zeros(self.max_len, dtype=int)
        self.right_cum_dist = np.zeros(self.max_len, dtype=int)
        for i_val in range(len(values)):
            cur_span_choice = spans[i_val]
            cur_val = values[i_val]
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
        while self._optimize_round():
            pass

    def _optimize_round(self) -> bool:
        has_changed: bool = False
        for i_val, cur_val in enumerate(self.values):
            # self.check_integrity()
            cur_span_choice = self.spans[i_val]
            if len(cur_span_choice) == 0:
                chosen_side = self.cur_solution[i_val]
                if chosen_side == SIDE_LEFT:
                    delta_energy = self.energy_move(self.left_cum_dist, len(cur_val), 0)
                    delta_energy += self.energy_move(self.right_cum_dist, 0, len(cur_val))
                    if delta_energy < -1e-8:
                        self.left_cum_dist[:len(cur_val)] -= 1
                        self.right_cum_dist[:len(cur_val)] += 1
                        self.cur_solution[i_val] = SIDE_RIGHT
                        has_changed = True
                elif chosen_side == SIDE_RIGHT:
                    delta_energy = self.energy_move(self.right_cum_dist, len(cur_val), 0)
                    delta_energy += self.energy_move(self.left_cum_dist, 0, len(cur_val))
                    if delta_energy < -1e-8:
                        self.left_cum_dist[:len(cur_val)] += 1
                        self.right_cum_dist[:len(cur_val)] -= 1
                        self.cur_solution[i_val] = SIDE_LEFT
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
                delta_E = (self.energy_move(self.left_cum_dist, i_start, j_start) +
                           self.energy_move(self.right_cum_dist, i_end, j_end))
                if delta_E - best_choice[1] > -1e-8:
                    continue
                best_choice = (j_choice,  delta_E)
            if best_choice[1] < 0:
                has_changed = True
                self.set_span(i_val, best_choice[0])
        return has_changed

    @property
    def energy(self) -> float:
        left_energy = np.sum(np.log(self.left_cum_dist+1))
        right_energy = np.sum(np.log(self.right_cum_dist+1))
        return left_energy + right_energy

    def energy_move(self, dist, len_src, len_dst):
        if len_dst > len_src:
            return np.sum(np.log((dist[len_src:len_dst]+2) /
                                 (dist[len_src:len_dst]+1)))
        elif len_dst < len_src:
            assert np.all(dist[len_dst:len_src] > 0), (dist[len_dst:len_src], len_src, len_dst)
            return np.sum(np.log((dist[len_dst:len_src] /
                                  (dist[len_dst:len_src]+1))))
        return 0

    def check_integrity(self):
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
    def energy_from_values(values):
        length_array = np.array([len(v) for v in values])
        lengths, counts = np.unique(length_array, return_counts=True)
        cum_len = np.zeros(np.max(lengths))
        for i in range(len(lengths)):
            cur_len = lengths[i]
            cur_counts = counts[i]
            cum_len[:cur_len] += cur_counts
        return np.sum(np.log(cum_len + 1))

    def set_span(self, i_val, i_span) -> None:
        old_span = self.spans[i_val][self.cur_solution[i_val]]
        new_span = self.spans[i_val][i_span]

        # First remove old ones
        if old_span[0] > new_span[0]:
            self.left_cum_dist[new_span[0]:old_span[0]] -= 1
        else:
            self.left_cum_dist[old_span[0]:new_span[0]] += 1

        old_right, new_right = len(self.values[i_val]) - np.array([old_span[1], new_span[1]])
        if old_right > new_right:
            self.right_cum_dist[new_right:old_right] -= 1
        else:
            self.right_cum_dist[old_right:new_right] += 1

        self.cur_solution[i_val] = i_span

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
