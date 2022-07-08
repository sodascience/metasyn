"""Utilities for disclosure control."""

import numpy as np


def get_cbs_bounds(values, n_avg: int=5) -> tuple[float, float]:
    """Get disclosure control bounds."""
    sorted_values = np.sort(values)
    sum_low = 0
    sum_high = 0
    for i_slice in range(n_avg):
        spliced_values = sorted_values[i_slice::n_avg]
        delta_avg = (spliced_values[-1]-spliced_values[0])/(len(spliced_values)-1)
        delta_avg /= n_avg
        low_est = spliced_values[0] - (i_slice+0.5)*delta_avg
        high_est = spliced_values[-1] + (n_avg-i_slice-0.5)*delta_avg
        sum_low += low_est
        sum_high += high_est
    return sum_low/n_avg, sum_high/n_avg
