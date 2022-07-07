import numpy as np


def get_cbs_bounds(values, n_avg=5):
    sorted_values = np.sort(values)
    sum_low = 0
    sum_high = 0
    for i in range(n_avg):
        spliced_values = sorted_values[i::n_avg]
        delta_avg = (spliced_values[-1]-spliced_values[0])/(len(spliced_values)-1)
        delta_avg /= n_avg
        low_est = spliced_values[0] - (i+0.5)*delta_avg
        high_est = spliced_values[-1] + (n_avg-i-0.5)*delta_avg
        sum_low += low_est
        sum_high += high_est
    return sum_low/n_avg, sum_high/n_avg
