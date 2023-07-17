"""Module for the base distribution and the scipy distribution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable, Sequence, Union, Optional

import numpy as np
import pandas as pd
import polars as pl


class BaseDistribution(ABC):
    """Abstract base class to define a distribution.

    All distributions should be derived from this class, and the following
    methods need to be implemented: _fit, draw, to_dict.
    """

    implements: str = "unknown"
    var_type: str = "unknown"
    provenance: str = "builtin"
    privacy: str = "none"
    is_unique: bool = False

    @classmethod
    def fit(cls, series: Union[Sequence, pl.Series],
            *args, **kwargs) -> BaseDistribution:
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
        pd_series = cls._to_series(series)
        if len(pd_series) == 0:
            return cls.default_distribution()
        return cls._fit(pd_series, *args, **kwargs)

    @staticmethod
    def _to_series(values: Union[Sequence, pl.Series]):
        if isinstance(values, pl.Series):
            series = values.drop_nulls()
        elif isinstance(values, pd.Series):
            series = pl.Series(values).drop_nulls()  # pylint: disable=assignment-from-no-return
        else:
            series_array = np.array(values)
            series_array = series_array[~np.isnan(series_array)]
            series = pl.Series(series_array)
        return series

    @classmethod
    @abstractmethod
    def _fit(cls, values: pl.Series) -> BaseDistribution:
        """See fit method, but does not need to deal with NA's."""

    @abstractmethod
    def draw(self) -> object:
        """Draw a random element from the fitted distribution."""

    def draw_reset(self) -> None:
        """Reset the drawing of elements to start again."""

    def __str__(self) -> str:
        """Return an easy to read formatted string for the distribution."""
        params_formatted = "\n".join(
            f"\t- {param}: {value}" for param,
            value in self._param_dict().items()
        )
        return (
            f"- Type: {self.implements}\n"
            f"- Provenance: {self.provenance}\n"
            f"- Parameters:\n"
            f"{params_formatted}\n"
        )

    @abstractmethod
    def _param_dict(self):
        """Get dictionary with the parameters of the distribution."""

    def to_dict(self) -> dict:
        """Convert the distribution to a dictionary."""
        return {
            "implements": self.implements,
            "provenance": self.provenance,
            "class_name": self.__class__.__name__,
            "parameters": deepcopy(self._param_dict()),
        }

    @classmethod
    @abstractmethod
    def _param_schema(cls):
        """Get schema for the parameters of the distribution."""

    @classmethod
    def schema(cls) -> dict:
        """Create sub-schema to validate GMF file."""
        return {
            "type": "object",
            "properties": {
                "implements": {"const": cls.implements},
                "provenance": {"const": cls.provenance},
                "class_name": {"const": cls.__name__},
                "parameters": {
                    "type": "object",
                    "properties": cls._param_schema(),
                    "required": list(cls.default_distribution()._param_dict())
                }
            },
            "required": ["implements", "provenance", "class_name", "parameters"]
        }

    @classmethod
    def from_dict(cls, dist_dict: dict) -> BaseDistribution:
        """Create a distribution from a dictionary."""
        return cls(**dist_dict["parameters"])

    def information_criterion(self, values: Iterable) -> float:  # pylint: disable=unused-argument
        """Get the AIC value for a particular set of values.

        Parameters
        ----------
        values: array_like
            Values to determine the AIC value of.
        """
        return 0.0

    @classmethod
    def matches_name(cls, name: str) -> bool:
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
        assert cls.implements != "unknown", f"Internal error in class {cls.__name__}"
        return name in (cls.implements.split(".")[1],
                        cls.implements,
                        cls.__name__,
                        )

    @classmethod
    @abstractmethod
    def default_distribution(cls) -> BaseDistribution:
        """Get a distribution with default parameters."""
        return cls()


def metadist(
        implements: Optional[str] = None,
        provenance: Optional[str] = None,
        var_type: Optional[str] = None,
        is_unique: Optional[bool] = None,
        privacy: Optional[str] = None):
    """Decorate class to create a distribution with the right properties.

    Parameters
    ----------
    implements:
        The distribution ID that it implements, e.g. core.uniform, core.regex.
    provenance:
        Where the distribution came from, which package/plugin implemented it.
    var_type:
        Variable type of the distribution, e.g. continuous, categorical, string.
    is_unique:
        Whether the distribution is unique or not.
    privacy:
        Privacy class/implementation of the distribution.

    Returns
    -------
    cls:
        Class with the appropriate class variables.
    """
    def _wrap(cls):
        if implements is not None:
            cls.implements = implements
        if provenance is not None:
            cls.provenance = provenance
        if var_type is not None:
            cls.var_type = var_type
        if is_unique is not None:
            cls.is_unique = is_unique
        if privacy is not None:
            cls.privacy = privacy
        return cls
    return _wrap


class ScipyDistribution(BaseDistribution):
    """Base class for numerical Scipy distributions.

    This base class makes it easy to implement new numerical
    distributions. One could also use this base class for non-scipy
    distributions, in which case the distribution class should implement
    logpdf, rvs and fit methods.
    """

    @property
    def n_par(self) -> int:
        """int: Number of parameters for distribution."""
        return len(self.par)

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
        if len(values) == 0:
            return cls.default_distribution()
        param = cls.dist_class.fit(values)
        return cls(*param)

    def _param_dict(self):
        return self.par

    def draw(self):
        return self.dist.rvs()

    def information_criterion(self, values):
        vals = self._to_series(values)
        if len(vals) == 0:
            return 2 * self.n_par
        return self._information_criterion(vals)

    def _information_criterion(self, values):
        return 2 * self.n_par - 2 * np.sum(self.dist.logpdf(values))
