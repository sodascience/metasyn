"""Module for distribution trees.

These are used to find/fit distributions that are available. See setup.py on how the
builtin distribution tree is registered.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Union
from typing import Type, Any, Optional
import warnings
import inspect
import pkg_resources

import polars as pl
import numpy as np

from metasynth.distribution.base import BaseDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution,\
    PoissonDistribution, UniqueKeyDistribution
from metasynth.distribution.continuous import UniformDistribution,\
    NormalDistribution, LogNormalDistribution, TruncatedNormalDistribution,\
    ExponentialDistribution
from metasynth.distribution.categorical import MultinoulliDistribution
from metasynth.distribution.regex.base import RegexDistribution,\
    UniqueRegexDistribution
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.datetime import UniformDateDistribution,\
    UniformTimeDistribution, UniformDateTimeDistribution


class BaseDistributionTree():
    """Class that encapsulates a set of distributions.

    It has a property {var_type}_distributions for every var_type.
    """

    def __init__(self, **kwargs):
        # Perform internal consistency check.
        for var_type in self.all_var_types:
            for dist in self.get_dist_list(var_type):
                assert dist.var_type == var_type, (f"Error: Distribution tree is inconsistent for "
                                                   f"{dist}.")
        self.privacy_kwargs = kwargs

    @property
    @abstractmethod
    def discrete_distributions(self) -> List[Type[BaseDistribution]]:
        """Get the integer distributions."""

    @property
    @abstractmethod
    def continuous_distributions(self) -> List[Type[BaseDistribution]]:
        """Get continuous distributions."""

    @property
    @abstractmethod
    def categorical_distributions(self) -> List[Type[BaseDistribution]]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def string_distributions(self) -> List[Type[BaseDistribution]]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def date_distributions(self) -> List[Type[BaseDistribution]]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def time_distributions(self) -> List[Type[BaseDistribution]]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def datetime_distributions(self) -> List[Type[BaseDistribution]]:
        """Get categorical distributions."""

    def get_dist_list(self, var_type: str) -> List[Type[BaseDistribution]]:
        """Get all distributions for a certain variable type.

        Parameters
        ----------
        var_type:
            Variable type to get the distributions for.

        Returns
        -------
        list[Type[BaseDistribution]]:
            List of distributions with that variable type.
        """
        prop_str = var_type + "_distributions"
        if not hasattr(self, prop_str):
            raise ValueError(f"Unknown variable type '{var_type}' detected.")
        return getattr(self, prop_str)

    def fit(self, series: pl.Series, var_type: str,
            unique: Optional[bool]=False) -> BaseDistribution:
        """Fit a distribution to a series.

        Search for the distirbution within all available distributions in the tree.

        Parameters
        ----------
        series:
            Series to fit a distribution to.
        var_type:
            Variable type of the series.
        unique:
            Whether the variable should be unique or not.

        Returns
        -------
        BaseDistribution:
            Distribution fitted to the series.
        """
        dist_list = self.get_dist_list(var_type)
        if len(dist_list) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'")
        dist_instances = [d.fit(series, **self.privacy_kwargs) for d in dist_list]
        dist_aic = [d.information_criterion(series) for d in dist_instances]
        i_best_dist = np.argmin(dist_aic)
        warnings.simplefilter("always")
        if dist_instances[i_best_dist].is_unique and unique is None:
            warnings.warn(f"\nVariable {series.name} seems unique, but not set to be unique.\n"
                          "Set the variable to be either unique or not unique to remove this "
                          "warning.\n")
        if unique is None:
            unique = False

        dist_aic = [dist_aic[i] for i in range(len(dist_aic))
                    if dist_instances[i].is_unique == unique]
        dist_instances = [d for d in dist_instances if d.is_unique == unique]
        if len(dist_instances) == 0:
            raise ValueError(f"No available distributions for variable '{series.name}'"
                             f" with variable type '{var_type}' "
                             f"that have unique == {unique}.")
        return dist_instances[np.argmin(dist_aic)]

    @property
    def all_var_types(self) -> List[str]:
        """Return list of available variable types."""
        return [p[:-14] for p in dir(self.__class__)
                if isinstance(getattr(self.__class__, p), property) and p.endswith("_distributions")
                ]

    def find_distribution(self, dist_name: str) -> tuple[Type[BaseDistribution], dict[str, Any]]:
        """Find a distribution and fit keyword arguments from a name.

        This allows us to use 'faker.city' to generate a faker instance that generates cities.

        Parameters
        ----------
        dist_name:
            Name of the distribution, such as faker.city, DiscreteUniformDistribution or normal.

        Returns
        -------
        tuple[Type[BaseDistribution], dict[str, Any]]:
            A distribution and the arguments to create an instance.
        """
        for var_type in self.all_var_types:
            for dist_class in self.get_dist_list(var_type):
                if dist_class.is_named(dist_name):
                    return dist_class, dist_class.fit_kwargs(dist_name)
        raise ValueError(f"Cannot find distribution with name '{dist_name}'.")

    def fit_distribution(self, dist: Union[str, Type[BaseDistribution], BaseDistribution],
                         series: pl.Series, **fit_kwargs) -> BaseDistribution:
        """Fit a specific distribution to a series.

        In contrast the fit method, this needs a supplied distribution(type).

        Parameters
        ----------
        dist:
            Distribution to fit (if it is not already fitted).
        series:
            Series to fit the distribution to
        fit_kwargs:
            Extra fitting parameters that are specific to the distribution.

        Returns
        -------
        BaseDistribution:
            Fitted distribution.
        """
        dist_instance = None
        fit_kwargs.update(self.privacy_kwargs)

        if isinstance(dist, str):
            dist_class, new_fit_kwargs = self.find_distribution(dist)
            fit_kwargs.update(new_fit_kwargs)
            dist_instance = dist_class.fit(series, **fit_kwargs)
        elif inspect.isclass(dist) and issubclass(dist, BaseDistribution):
            dist_instance = dist.fit(series, **fit_kwargs)
        if isinstance(dist, BaseDistribution):
            dist_instance = dist

        if dist_instance is None:
            raise TypeError(
                f"Distribution with type {type(dist)} is not a BaseDistribution")

        return dist_instance

    def from_dict(self, var_dict: dict[str, Any]) -> BaseDistribution:
        """Create a distribution from a dictionary.

        Parameters
        ----------
        var_dict:
            Variable dictionary that includes the distribution properties.

        Returns
        -------
        BaseDistribution:
            Distribution representing the dictionary.
        """
        for dist_class in self.get_dist_list(var_dict["type"]):
            if dist_class.is_named(var_dict["distribution"]["name"]):
                return dist_class.from_dict(var_dict["distribution"])
        raise ValueError(f"Cannot find distribution with name '{var_dict['distribution']['name']}'"
                         f"and type '{var_dict['type']}'.")


class BuiltinDistributionTree(BaseDistributionTree):
    """Distribution tree that includes the builtin distributions."""

    @property
    def discrete_distributions(self) -> List[type]:
        return [DiscreteUniformDistribution, PoissonDistribution, UniqueKeyDistribution]

    @property
    def continuous_distributions(self) -> List[type]:
        return [UniformDistribution, NormalDistribution, LogNormalDistribution,
                TruncatedNormalDistribution, ExponentialDistribution]

    @property
    def categorical_distributions(self) -> List[type]:
        return [MultinoulliDistribution]

    @property
    def string_distributions(self) -> List[type]:
        return [RegexDistribution, UniqueRegexDistribution, FakerDistribution]

    @property
    def date_distributions(self) -> List[type]:
        return [UniformDateDistribution]

    @property
    def time_distributions(self) -> List[type]:
        return [UniformTimeDistribution]

    @property
    def datetime_distributions(self) -> List[type]:
        return [UniformDateTimeDistribution]


def get_disttree(target: Optional[Union[str, type, BaseDistributionTree]]=None, **kwargs
                 ) -> BaseDistributionTree:
    """Get a distribution tree.

    Parameters
    ----------
    target:
        Directive to get the distribution tree.

    Returns
    -------
    BaseDistributionTree:
        Distribution tree.
    """
    if target is None:
        target = "builtin"
    if isinstance(target, BaseDistributionTree):
        return target
    if isinstance(target, type):
        return target()

    all_disttrees = {
        entry.name: entry
        for entry in pkg_resources.iter_entry_points("metasynth.disttree")
    }
    try:
        return all_disttrees[target].load()(**kwargs)
    except KeyError as exc:
        raise ValueError(f"Cannot find distribution tree with name '{target}'.") from exc
