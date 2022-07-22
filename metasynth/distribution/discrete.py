"""Module with discrete distributions."""

from typing import Set

import numpy as np
import pandas as pd
from scipy.stats import randint, poisson

from metasynth.distribution.base import ScipyDistribution, DiscreteDistribution


class DiscreteUniformDistribution(ScipyDistribution, DiscreteDistribution):
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

    aliases = ["DiscreteUniformDistribution", "discrete_uniform"]

    dist_class = randint

    def __init__(self, low: int, high: int):
        self.par = {"low": low, "high": high}
        self.dist = self.dist_class(low=low, high=high)

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        if isinstance(vals, pd.Series):
            vals = vals.values.astype(int)
        return 2*self.n_par - 2*np.sum(self.dist.logpmf(vals))

    @classmethod
    def _fit(cls, values):
        param = {"low": np.min(values), "high": np.max(values)+1}
        return cls(**param)

    @classmethod
    def _example_distribution(cls):
        return cls(0, 10)


class PoissonDistribution(ScipyDistribution, DiscreteDistribution):
    """Poisson distribution."""

    aliases = ["PoissonDistribution", "poisson"]
    dist_class = poisson

    def __init__(self, mu: float):
        self.par = {"mu": mu}
        self.dist = self.dist_class(mu=mu)

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        if isinstance(vals, pd.Series):
            vals = vals.values.astype(int)
        return 2*self.n_par - 2*np.sum(self.dist.logpmf(vals))

    @classmethod
    def _fit(cls, values):
        return cls(np.mean(values))

    @classmethod
    def _example_distribution(cls):
        return cls(0.5)


class UniqueKeyDistribution(ScipyDistribution, DiscreteDistribution):
    """Integer distribution with unique keys.

    Discrete distribution that ensures the uniqueness of the drawn values.

    Parameters
    ----------
    low: int
        Minimum value for the keys.
    consecutive: int
        1 if keys are consecutive and increasing, 0 otherwise.
    """

    aliases = ["UniqueKeyDistribution", "unique_key"]
    is_unique = True

    def __init__(self, low: int, consecutive: int):
        self.par = {"low": low, "consecutive": consecutive}
        self.last_key = low - 1
        self.key_set: Set[int] = set()

    @classmethod
    def _fit(cls, values):
        low = np.min(values)
        high = np.max(values)+1
        if len(values) == high-low and np.all(values == np.arange(low, high)):
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

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        if np.min(values) < self.low:
            return 3+999*len(values)

        # If the values are not unique the fit is extremely bad.
        if len(set(vals)) != len(vals):
            return 3+999*len(values)

        low = np.min(vals)
        high = np.max(vals)+1

        if self.consecutive == 1:
            # Check if the values are truly
            if len(vals) == high-low and np.all(vals == np.arange(low, high)):
                return 3
            return 3+999*len(values)

        n_choice = high - low

        # Probabilities go up like 1/n, 1/(n-1), 1/(n-2), ..., 1/2, 1
        return 5 - 2*np.sum(np.log(1/np.arange(n_choice, n_choice-len(values), -1)))

    @classmethod
    def _example_distribution(cls):
        return cls(0, 0)
