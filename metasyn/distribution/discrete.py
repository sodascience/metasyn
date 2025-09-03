"""Module implementing discrete distributions."""

from typing import Set

import numpy as np
from scipy.stats import poisson, randint

from metasyn.distribution.base import (
    BaseConstantDistribution,
    BaseDistribution,
    ScipyDistribution,
    metadist,
)
from metasyn.distribution.continuous import NormalDistribution, TruncatedNormalDistribution


@metadist(implements="core.poisson", var_type="discrete")
class PoissonDistribution(ScipyDistribution):
    """Poisson distribution.

    Parameters
    ----------
    rate:
        Rate (mean) of the poisson distribution.

    Examples
    --------
    >>> PoissonDistribution(rate=3.5)
    """

    dist_class = poisson

    def __init__(self, rate: float):
        self.par = {"rate": rate}
        self.dist = self.dist_class(mu=rate)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))

    @classmethod
    def _fit(cls, values):
        mean_values = values.mean()
        mean_values = mean_values if mean_values >= 0 else 0
        return cls(mean_values)

    @classmethod
    def default_distribution(cls):
        return cls(0.5)

    @classmethod
    def _param_schema(cls):
        return {
            "rate": {"type": "number"},
        }


@metadist(implements="core.unique_key", var_type="discrete", unique=True)
class UniqueKeyDistribution(BaseDistribution):
    """Unique key distribution for identifiers.

    Discrete distribution that ensures the uniqueness of the drawn values.

    Parameters
    ----------
    lower:
        Minimum value for the keys.
    consecutive:
        True if keys are consecutive and increasing, False otherwise.

    Examples
    --------
    >>> UniqueKeyDistribution(lower=0, consecutive=True)
    """

    def __init__(self, lower: int, consecutive: bool):
        self.par = {"lower": lower, "consecutive": consecutive}
        self.last_key = lower - 1
        self.key_set: Set[int] = set()

    def _param_dict(self):
        return self.par

    def __getattr__(self, attr: str):
        """Get attribute for easy access to parameters.

        Parameters
        ----------
        attr:
            Attribute to retrieve. If the attribute is a parameter
            name, then retrieve that parameter, otherwise use the default
            implementation for getting the attribute.

        Returns
        -------
        object:
            Parameter or attribute.
        """
        if attr != "par" and attr in self.par:
            return self.par[attr]
        return object.__getattribute__(self, attr)

    @classmethod
    def _fit(cls, values):
        lower = values.min()
        high = values.max() + 1
        if len(values) == high-lower and np.all(values.to_numpy() == np.arange(lower, high)):
            return cls(lower, True)
        return cls(lower, False)

    def draw_reset(self):
        self.last_key = self.lower - 1
        self.key_set = set()

    def draw(self):
        if self.consecutive == 1:
            self.last_key += 1
            return self.last_key

        while True:
            random_number = np.random.randint(self.lower,
                                              self.lower+2*len(self.key_set)+2)
            if random_number not in self.key_set:
                self.key_set.add(random_number)
                return random_number

    def information_criterion(self, values):
        values = self._to_series(values)
        if len(values) == 0:
            return 2 * 2

        if values.min() < self.lower:
            return 2*np.log(len(values))+999*len(values)

        # If the values are not unique the fit is extremely bad.
        if len(set(values)) != len(values):
            return 2*np.log(len(values))+999*len(values)

        lower = values.min()
        high = values.max()+1

        if self.consecutive == 1:
            # Check if the values are truly consecutive
            if len(values) == high-lower and np.all(values.to_numpy() == np.arange(lower, high)):
                return 2*np.log(len(values))
            return 2*np.log(len(values))+999*len(values)

        n_choice = high - lower

        # Probabilities go up like 1/n, 1/(n-1), 1/(n-2), ..., 1/2, 1
        return (3*np.log(len(values))
                - 2*np.sum(np.log(1/np.arange(n_choice, n_choice-len(values), -1))))

    @classmethod
    def default_distribution(cls):
        return cls(0, False)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "integer"},
            "consecutive": {"type": "boolean"},
        }

