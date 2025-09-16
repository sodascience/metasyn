"""Module serving as the basis for all metasyn distributions.

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
from typing import Any, Optional, Sequence, Union

import numpy as np
import polars as pl
from numpy import typing as npt

from metasyn.privacy import BasePrivacy


def convert_to_series(values: Union[npt.NDArray, pl.Series]) -> pl.Series:
    """Convert list or pandas series to polars series."""
    if not isinstance(values, (pl.Series, np.ndarray)):
        values = pl.Series(values)
    if isinstance(values, pl.Series):
        series = values.drop_nulls()
    else:
        series_array = np.array(values)
        series_array = series_array[~np.isnan(series_array)]
        series = pl.Series(series_array)
    return series

class BaseFitter(ABC):
    """Base class for fitters."""

    distribution: type[BaseDistribution]
    privacy_type: str = "none"
    version: str = "1.0"
    var_type: Union[str, list[str]] = "unknown"

    def __init__(self, privacy: BasePrivacy):
        self.privacy = privacy

    # @abstractmethod
    def fit(self, values: Union[npt.NDArray, pl.Series]) -> BaseDistribution:
        pl_series = convert_to_series(values)
        if len(pl_series) == 0:
            return self.distribution.default_distribution()
        return self._fit(pl_series)

    @abstractmethod
    def _fit(self, series: pl.Series) -> BaseDistribution:
        raise NotImplementedError

    @classmethod
    def matches_name(cls, name: str) -> bool:
        """Check whether the name matches the fitter.

        Parameters
        ----------
        name: str
            Name to match to the fitter.

        Returns
        -------
        bool:
            Whether the name matches.
        """
        assert cls.distribution.name != "unknown", f"Internal error in class {cls.__name__}"
        return name in (cls.distribution.name.split(".")[1],
                        cls.distribution.name,
                        cls.__name__,
                        )



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

    name: str = "unknown"
    """The identifier for the implemented distribution"""
    var_type: Union[str, Sequence[str]] = "unknown"
    """The variable type of the distribution"""
    unique: bool = False
    """Whether the distribution creates only unique values"""
    version: str = "1.0"
    """Version of the implemented distribution"""

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
            f"- Type: {self.name}\n"
            f"- Parameters:\n"
            f"{self._params_formatted}\n"
        )

    @abstractmethod
    def _param_dict(self):
        """Get dictionary with the parameters of the distribution."""

    def to_dict(self) -> dict:
        """Convert the distribution to a dictionary."""
        return {
            "name": self.name,
            "version": self.version,
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
                "name": {"const": cls.name},
                "version": {"type": "string"},
                "class_name": {"const": cls.__name__},
                "unique": {"const": cls.unique},
                "parameters": {
                    "type": "object",
                    "properties": cls._param_schema(),
                    "required": list(cls.default_distribution()._param_dict())
                }
            },
            "required": ["name", "class_name", "parameters"]
        }

    @classmethod
    def from_dict(cls, dist_dict: dict) -> BaseDistribution:
        """Create a distribution from a dictionary."""
        return cls(**dist_dict["parameters"])

    def information_criterion(self, values: Union[pl.Series, npt.NDArray]) -> float: # noqa: ARG002
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
        assert cls.name != "unknown", f"Internal error in class {cls.__name__}"
        return name in (cls.name.split(".")[1],
                        cls.name,
                        cls.__name__,
                        )

    @classmethod
    @abstractmethod
    def default_distribution(cls) -> BaseDistribution:
        """Get a distribution with default parameters."""
        return cls()

    def draw_list(self, n: int) -> list:
        """Draw a list of values from the distribution.

        Parameters
        ----------
        n:
            Number of items to draw from the distribution.

        Raises
        ------
        NotImplementedError:
            If the distribution hasn't implemented a draw_list.

        Returns
        -------
            List of values.
        """
        raise NotImplementedError()

def metadist(
        name: Optional[str] = None,
        var_type: Optional[Union[str, list[str]]] = None,
        unique: Optional[bool] = None,
        version: Optional[str] = None):
    """Decorate class to create a distribution with the right properties.

    Parameters
    ----------
    name:
        Name that identifies the distribution uniquely, e.g. core.uniform, core.regex.
        The name should use a period (.) so that the first part is the namespace (e.g. core),
        and the second part the name of the distribution.
    var_type:
        Variable type of the distribution, e.g. continuous, categorical, string.
    unique:
        Whether the distribution is unique or not.
    version:
        Version of the distribution. Increment this to ensure that compatibility is
        properly handled.

    Returns
    -------
    cls:
        Class with the appropriate class variables.
    """
    def _wrap(cls):
        if name is not None:
            cls.name = name
        if var_type is not None:
            cls.var_type = var_type
        if unique is not None:
            cls.unique = unique
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
    name:
        {cls.name}
    unique:
        {cls.unique}
    version:
        {cls.version}
    var_type:
        {cls.var_type}
    """
        return cls

    return _wrap


def metafit(
        distribution: Optional[type[BaseDistribution]] = None,
        var_type: Optional[Union[str, list[str]]] = None,
        version: Optional[str] = None,
        privacy_type: Optional[str] = None):
    """Decorate class to create a fitter with the correct class attributes.

    Parameters
    ----------
    distribution:
        Class that the fitter will return after a succesful fit.
    var_type:
        Variable type(s) that the fitter implements, e.g. continuous, categorical, string.
    version:
        Version of the fitter. Increment this to ensure that compatibility is
        properly handled.
    privacy_type:
        Privacy class/implementation of the fitter.

    Returns
    -------
    cls:
        Class with the appropriate class variables.
    """
    def _wrap(cls):
        if distribution is not None:
            cls.distribution = distribution
        if var_type is not None:
            cls.var_type = var_type
        # if unique is not None:
            # cls.unique = unique
        if privacy_type is not None:
            cls.privacy_type = privacy_type
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
    dist_class:
        {cls.distribution}
    version:
        {cls.version}
    var_type:
        {cls.var_type}
    privacy:
        {cls.privacy_type}
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

    def _param_dict(self):
        return self.par

    def draw(self):
        val = self.dist.rvs()
        if self.var_type == "discrete":
            return int(val)
        return val

    def draw_list(self, n: int) -> list:
        values = self.dist.rvs(n)
        if self.var_type == "discrete":
            values = values.astype(np.int64)
        return list(values)

    def information_criterion(self, values):
        vals = convert_to_series(values)
        if len(vals) == 0:
            return 2 * self.n_par
        return self._information_criterion(vals)

    def _information_criterion(self, values):
        return np.log(len(values)) * self.n_par - 2 * np.sum(self.dist.logpdf(values))


class ScipyFitter(BaseFitter):
    """Base fitter for scipy distributions."""

    def _fit(self, series):
        param = self.distribution.scipy_class.fit(series)  # type: ignore  # All derived classes have dist_class
        return self.distribution(*param)



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

    def __new__(cls, *args, **kwargs):  # noqa
        instance = super().__new__(cls)
        instance.key_set: set[Any] = set()
        return instance

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

    def information_criterion(self, values): # noqa: ARG002
        return 9999999


