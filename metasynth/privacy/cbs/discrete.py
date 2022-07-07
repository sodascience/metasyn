import numpy as np

from metasynth.distribution.discrete import DiscreteUniformDistribution,\
    PoissonDistribution, UniqueKeyDistribution


class CbsDiscreteUniform(DiscreteUniformDistribution):
    @classmethod
    def _fit(cls, values, n_avg: int=5):
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
        return cls(round(sum_low/n_avg), round(sum_high/n_avg))


class CbsPoisson(PoissonDistribution):
    pass


class CbsUniqueKey(UniqueKeyDistribution):
    @classmethod
    def _fit(cls, values, n_avg: int=5):
        orig_dist = super()._fit(values)
        if orig_dist.consecutive == 1:
            return cls(np.random.randint(2*n_avg)-n_avg, orig_dist.consecutive)
        uniform_dist = CbsDiscreteUniform._fit(values, n_avg=n_avg)
        return cls(uniform_dist.low, orig_dist.consecutive)
