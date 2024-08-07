"""
Module serving as the basis for all metasyn distributions.

The base module contains the ``BaseDistribution`` class,
which is the base class for all distributions.
It also contains the ``ScipyDistribution`` class,
which is a specialized base class for distributions that are built on top of
SciPy's statistical distributions.

Additionally it contains the ``UniqueDistributionMixin`` class,
which is a mixin class that can be used to make a distribution unique
(i.e., one that does not contain duplicate values).

Finally it contains the ``metadist()`` decorator,
which is used to set the class attributes of a distribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import polars as pl
from numpy import inf
from numpy import typing as npt


class BaseDistribution(ABC):
    """Abstract base class to define a distribution.

    All distributions should be derived from this class, and should implement
    the following methods:
    :meth:`~_fit`,
    :meth:`~draw`,
    :meth:`~_param_dict`,
    :meth:`~_param_schema`,
    :meth:`~default_distribution`
    and ``__init__``.
    """

    implements: str = "unknown"
    """The identifier for the implemented distribution"""
    var_type: str = "unknown"
    """The variable type of the distribution"""
    provenance: str = "builtin"
    """Which plugin provides the distribution or builtin"""
    privacy: str = "none"
    """The type of privacy it implements"""
    unique: bool = False
    """Whether the distribution creates only unique values"""
    version: str = "1.0"
    """Version of the implemented distribution"""

    @classmethod
    def fit(cls, series: Union[pl.Series, npt.NDArray],
            *args, **kwargs) -> BaseDistribution:
        """Fit the distribution to the series.

        Parameters
        ----------
        series: polars.Series
            Data to fit the distribution to.

        Returns
        -------
        BaseDistribution:
            Fitted distribution.
        """
        pl_series = cls._to_series(series)
        if len(pl_series) == 0:
            return cls.default_distribution()
        return cls._fit(pl_series, *args, **kwargs)

    @staticmethod
    def _to_series(values: Union[npt.NDArray, pl.Series]) -> pl.Series:
        if not isinstance(values, (pl.Series, np.ndarray)):
            values = pl.Series(values)
        if isinstance(values, pl.Series):
            series = values.drop_nulls()
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

    @property
    def _params_formatted(self) -> str:
        return "\n".join(
            f"\t- {param}: {value}" for param,
            value in self._param_dict().items()
        )

    def __str__(self) -> str:
        """Return an easy to read formatted string for the distribution."""
        return (
            f"- Type: {self.implements}\n"
            f"- Provenance: {self.provenance}\n"
            f"- Parameters:\n"
            f"{self._params_formatted}\n"
        )

    @abstractmethod
    def _param_dict(self):
        """Get dictionary with the parameters of the distribution."""

    def to_dict(self) -> dict:
        """Convert the distribution to a dictionary."""
        return {
            "implements": self.implements,
            "version": self.version,
            "provenance": self.provenance,
            "class_name": self.__class__.__name__,
            "unique": self.unique,
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
                "version": {"type": "string"},
                "provenance": {"const": cls.provenance},
                "class_name": {"const": cls.__name__},
                "unique": {"const": cls.unique},
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

    def information_criterion(self, values: Union[pl.Series, npt.NDArray]) -> float:  # pylint: disable=unused-argument
        """Get the BIC value for a particular set of values.

        Parameters
        ----------
        values: array_like
            Values to determine the BIC value of.
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
        var_type: Optional[Union[str, list[str]]] = None,
        unique: Optional[bool] = None,
        version: Optional[str] = None,
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
    unique:
        Whether the distribution is unique or not.
    privacy:
        Privacy class/implementation of the distribution.
    version:
        Version of the distribution. Increment this to ensure that compatibility is
        properly handled.

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
        if unique is not None:
            cls.unique = unique
        if privacy is not None:
            cls.privacy = privacy
        if version is not None:
            cls.version = version
        if cls.__doc__ is None:
            return cls
        cls.__doc__ = cls.__doc__.rstrip(" ")
        if not cls.__doc__.endswith("\n"):
            cls.__doc__ += "\n"
        cls.__doc__ += f"""
    Attributes
    ----------
    implements:
        {cls.implements}
    unique:
        {cls.unique}
    version:
        {cls.version}
    var_type:
        {cls.var_type}
    privacy:
        {cls.privacy}
    provenance:
        {cls.provenance}
    """
        return cls

    return _wrap


class ScipyDistribution(BaseDistribution):
    """Base class for numerical distributions using Scipy.

    This base class makes it easy to implement new numerical
    distributions. It can also be used for non-Scipy distributions,
    provided the distribution implements `logpdf`, `rvs` and `fit` methods.
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
        param = cls.dist_class.fit(values)  # type: ignore  # All derived classes have dist_class
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
        return np.log(len(values)) * self.n_par - 2 * np.sum(self.dist.logpdf(values))


@metadist(unique=True)
class UniqueDistributionMixin(BaseDistribution):
    """Mixin class to make unique version of base distributions.

    This mixin class can be used to extend base distribution classes, adding
    functionality that ensures generated values are unique. It overrides
    the `draw` method of the base class, adding a check to prevent duplicate
    values from being drawn. If a duplicate value is drawn, it retries up to
    1e5 times before raising a ValueError.

    The `UniqueDistributionMixin` is used in various unique metasyn distribution
    variations, such as `UniqueFakerDistribution` and `UniqueRegexDistribution`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_set: set = set()

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> object:
        n_retry = 0
        while n_retry < 1e5:
            new_val = super().draw()  # type: ignore
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val
            n_retry += 1
        raise ValueError(f"Failed to draw unique string after {n_retry} tries.")

    def information_criterion(self, values):
        return 9999999



class BaseConstantDistribution(BaseDistribution):
    """Base class for constant distribution.

    This base class makes it easy to implement new constant distributions
    for different variable types.
    """

    def __init__(self, value) -> None:
        self.value = value

    @property
    def n_par(self) -> int:
        """int: Number of parameters for distribution."""
        return 0

    @classmethod
    def _fit(cls, values: pl.Series) -> BaseDistribution:
        # if unique, just get that value
        if values.n_unique() == 1:
            return cls(values.unique()[0])

        # otherwise get most common value
        val = values.value_counts(sort=True)[0,0]
        return cls(val)

    def _param_dict(self):
        return {"value": self.value}

    def draw(self):
        return self.value

    def information_criterion(self, values):
        vals = self._to_series(values)
        return -inf if vals.n_unique() < 2 else inf
