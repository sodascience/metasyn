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


@metadist(implements="core.uniform", var_type="discrete")
class DiscreteUniformDistribution(ScipyDistribution):
    """Uniform discrete distribution.

    It differs from the floating point uniform distribution by
    being a discrete distribution instead.

    Parameters
    ----------
    lower: int
        Lower bound (inclusive) of the uniform distribution.
    upper: int
        Upper bound (exclusive) of the uniform distribution.

    Examples
    --------
    >>> DiscreteUniformDistribution(lower=3, upper=20)
    """

    dist_class = randint

    def __init__(self, lower: int, upper: int):
        self.par = {"lower": lower, "upper": upper}
        self.dist = self.dist_class(low=lower, high=upper)

    def _information_criterion(self, values):
        return np.log(len(values))*self.n_par - 2*np.sum(self.dist.logpmf(values))

    @classmethod
    def _fit(cls, values):
        param = {"lower": values.min(), "upper": values.max()+1}
        return cls(**param)

    @classmethod
    def default_distribution(cls):
        return cls(0, 10)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "integer"},
            "upper": {"type": "integer"},
        }

@metadist(implements="core.normal", var_type="discrete")
class DiscreteNormalDistribution(NormalDistribution):
    """Normal discrete distribution.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    sd: float
        Standard deviation of the normal distribution.

    Examples
    --------
    >>> DiscreteNormalDistribution(mean=2.4, sd=1.2)
    """

    def draw(self):
        return int(super().draw())

@metadist(implements="core.truncated_normal", var_type="discrete")
class DiscreteTruncatedNormalDistribution(TruncatedNormalDistribution):
    """Truncated normal discrete distribution.

    Parameters
    ----------
    lower: float
        Lower bound of the truncated normal distribution.
    upper: float
        Upper bound of the truncated normal distribution.
    mean: float
        Mean of the non-truncated normal distribution.
    sd: float
        Standard deviation of the non-truncated normal distribution.

    Examples
    --------
    >>> DiscreteTruncatedNormalDistribution(lower=1.2, upper=4.5, mean=2.3, sd=4.5)
    """

    def draw(self):
        return int(super().draw())


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
        return cls(values.mean())

    @classmethod
    def default_distribution(cls):
        return cls(0.5)

    @classmethod
    def _param_schema(cls):
        return {
            "rate": {"type": "number"},
        }


@metadist(implements="core.unique_key", var_type="discrete", unique=True)
class UniqueKeyDistribution(ScipyDistribution):
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
            random_number = np.random.randint(self.lower, self.lower+2*len(self.key_set)+2)
            if random_number not in self.key_set:
                self.key_set.add(random_number)
                return random_number

    def _information_criterion(self, values):
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

@metadist(implements="core.constant", var_type="discrete")
class DiscreteConstantDistribution(BaseConstantDistribution):
    """Constant discrete distribution.

    This class implements the constant distribution, so that it draws always
    the same value.

    Parameters
    ----------
    value: int
        Value that will be returned when drawn.

    Examples
    --------
    >>> DiscreteConstantDistribution(213456)
    """

    @classmethod
    def default_distribution(cls) -> BaseDistribution:
        return cls(0)

    @classmethod
    def _param_schema(cls):
        return {
            "value": {"type": "integer"}
        }
