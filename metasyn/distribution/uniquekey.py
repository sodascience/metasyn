"""Unique key distributions, often for primary key columns."""

from typing import Set, TypeVar

import numpy as np

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    convert_to_series,
    metadist,
    metafit,
)

UKeyT = TypeVar("UKeyT", bound="UniqueKeyDistribution")


@metadist(name="core.unique_key", var_type="discrete", unique=True)
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
        values = convert_to_series(values)
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
    def default_distribution(cls: type[UKeyT], var_type=None) -> UKeyT: # noqa: ARG003
        return cls(0, False)

    @classmethod
    def _param_schema(cls):
        return {
            "lower": {"type": "integer"},
            "consecutive": {"type": "boolean"},
        }

@metafit(distribution=UniqueKeyDistribution, var_type="discrete")
class UniqueKeyFitter(BaseFitter):
    """Fitter for unique key distribution."""

    distribution: type[UniqueKeyDistribution]

    def _fit(self, series) -> UniqueKeyDistribution:
        lower = series.min()
        high = series.max() + 1
        if len(series) == high-lower and np.all(series.to_numpy() == np.arange(lower, high)):
            return self.distribution(lower, True)
        return self.distribution(lower, False)
