"""Module for the base distribution and the scipy distribution."""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class BaseDistribution(ABC):
    """Abstract base class to define a distribution.

    All distributions should be derived from this class, and the following
    methods need to be implemented: _fit, draw, to_dict.
    """

    aliases = []

    @classmethod
    def fit(cls, series, *args, **kwargs):
        """Fit the distribution to the series.

        Parameters
        ----------
        series: pandas.Series
            Data to fit the distribution to.

        Returns
        -------
        BaseDistribution:
            Fitted distribution.
        """
        distribution = cls._fit(series.dropna(), *args, **kwargs)
        return distribution

    @classmethod
    @abstractmethod
    def _fit(cls, values):
        """See fit method, but does not need to deal with NA's."""

    @abstractmethod
    def draw(self):
        """Draw a random element from the fitted distribution."""

    def __str__(self):
        """Create a human readable string of the object."""
        return str(self.to_dict())

    @abstractmethod
    def to_dict(self):
        """Convert the distribution to a dictionary."""

    def information_criterion(self, values):  # pylint: disable=unused-argument, no-self-use
        """Get the AIC value for a particular set of values.

        Parameters
        ----------
        values: array_like
            Values to determine the AIC value of.
        """
        return 0.0

    @classmethod
    def is_named(cls, name):
        """Check whether the name matches the distribution.

        Parameters
        ----------
        name: str
            Name to match to the distribution.

        Returns
        -------
        bool:
            Whether the name matches.
        """
        return name in cls.aliases or name == type(cls).__name__ or name == cls.__name__

    @classmethod
    def fit_kwargs(cls, name):  # pylint: disable=unused-argument
        """Extra fitting arguments.

        Parameters
        ----------
        name: str
            Name to be matched.

        Returns
        -------
        dict:
            Keyword arguments extracted from the name.
        """
        return {}


class ScipyDistribution(BaseDistribution):
    """Base class for numerical Scipy distributions.

    This base class makes it easy to implement new numerical
    distributions. One could also use this base class for non-scipy
    distributions, in which case the distribution class should implement
    logpdf, rvs and fit methods.
    """

    @property
    def n_par(self):
        """int: Number of parameters for distribution."""
        return len(self.par)

    def __getattr__(self, attr):
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
        return super().__getattr__(attr)  # pylint: disable=no-member

    @classmethod
    def _fit(cls, values):
        param = cls.dist_class.fit(values[~np.isnan(values)])
        return cls(*param)

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": deepcopy(self.par),
        }

    def draw(self):
        return self.dist.rvs()

    def information_criterion(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpdf(vals+1e-7))
