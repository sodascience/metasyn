"""Module with discrete distributions."""

from typing import Set

import numpy as np
from scipy.stats import poisson, randint

from metasyn.distribution.base import ScipyDistribution, metadist
from metasyn.distribution.continuous import NormalDistribution, TruncatedNormalDistribution


@metadist(implements="core.discrete_uniform", var_type="discrete")
class DiscreteUniformDistribution(ScipyDistribution):
    """Integer uniform distribution.

    It differs from the floating point uniform distribution by
    being a discrete distribution instead.

    Parameters
    ----------
    low: int
        Lower bound (inclusive) of the uniform distribution.
    high: int
        Upper bound (exclusive) of the uniform distribution.
    """

    dist_class = randint

    def __init__(self, low: int, high: int):
        self.par = {"low": low, "high": high}
        self.dist = self.dist_class(low=low, high=high)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))

    @classmethod
    def _fit(cls, values):
        param = {"low": values.min(), "high": values.max()+1}
        return cls(**param)

    @classmethod
    def default_distribution(cls):
        return cls(0, 10)

    @classmethod
    def _param_schema(cls):
        return {
            "low": {"type": "integer"},
            "high": {"type": "integer"},
        }

@metadist(implements="core.discrete_normal", var_type="discrete")
class DiscreteNormalDistribution(NormalDistribution):
    """Normal distribution for integer type.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    std_dev: float
        Standard deviation of the normal distribution.
    """

    def draw(self):
        return int(super().draw())

@metadist(implements="core.discrete_truncated_normal", var_type="discrete")
class DiscreteTruncatedNormalDistribution(TruncatedNormalDistribution):
    """Truncated normal distribution for integer type.

    Parameters
    ----------
    lower_bound: float
        Lower bound of the truncated normal distribution.
    upper_bound: float
        Upper bound of the truncated normal distribution.
    mu: float
        Mean of the non-truncated normal distribution.
    sigma: float
        Standard deviation of the non-truncated normal distribution.
    """
    
    def draw(self):
        return int(super().draw())


@metadist(implements="core.poisson", var_type="discrete")
class PoissonDistribution(ScipyDistribution):
    """Poisson distribution."""

    dist_class = poisson

    def __init__(self, mu: float):
        self.par = {"mu": mu}
        self.dist = self.dist_class(mu=mu)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))

    @classmethod
    def _fit(cls, values):
        return cls(values.mean())

    @classmethod
    def default_distribution(cls):
        return cls(0.5)

    @classmethod
    def _param_schema(cls):
        return {
            "mu": {"type": "number"},
        }


@metadist(implements="core.unique_key", var_type="discrete", is_unique=True)
class UniqueKeyDistribution(ScipyDistribution):
    """Integer distribution with unique keys.

    Discrete distribution that ensures the uniqueness of the drawn values.

    Parameters
    ----------
    low: int
        Minimum value for the keys.
    consecutive: int
        1 if keys are consecutive and increasing, 0 otherwise.
    """

    def __init__(self, low: int, consecutive: int):
        self.par = {"low": low, "consecutive": consecutive}
        self.last_key = low - 1
        self.key_set: Set[int] = set()

    @classmethod
    def _fit(cls, values):
        low = values.min()
        high = values.max() + 1
        if len(values) == high-low and np.all(values.to_numpy() == np.arange(low, high)):
            return cls(low, 1)
        return cls(low, 0)

    def draw_reset(self):
        self.last_key = self.low - 1
        self.key_set = set()

    def draw(self):
        if self.consecutive == 1:
            self.last_key += 1
            return self.last_key

        while True:
            random_number = np.random.randint(self.low, self.low+2*len(self.key_set)+2)
            if random_number not in self.key_set:
                self.key_set.add(random_number)
                return random_number

    def _information_criterion(self, values):
        if values.min() < self.low:
            return 2*np.log(len(values))+999*len(values)

        # If the values are not unique the fit is extremely bad.
        if len(set(values)) != len(values):
            return 2*np.log(len(values))+999*len(values)

        low = values.min()
        high = values.max()+1

        if self.consecutive == 1:
            # Check if the values are truly consecutive
            if len(values) == high-low and np.all(values.to_numpy() == np.arange(low, high)):
                return 2*np.log(len(values))
            return 2*np.log(len(values))+999*len(values)

        n_choice = high - low

        # Probabilities go up like 1/n, 1/(n-1), 1/(n-2), ..., 1/2, 1
        return (3*np.log(len(values))
                - 2*np.sum(np.log(1/np.arange(n_choice, n_choice-len(values), -1))))

    @classmethod
    def default_distribution(cls):
        return cls(0, 0)

    @classmethod
    def _param_schema(cls):
        return {
            "low": {"type": "integer"},
            "high": {"type": "integer"},
        }
