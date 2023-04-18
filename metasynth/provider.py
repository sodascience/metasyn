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
try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore

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
from metasynth.privacy import BasePrivacy, NoPrivacy


class BaseDistributionProvider():
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
    def distributions(self) -> List[Type[BaseDistribution]]:
        """Get the all available distributions."""

    @property
    @abstractmethod
    def name(self):
        """Name of the distribution package."""

    @property
    @abstractmethod
    def version(self):
        """Version of the distribution package."""

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
        return [dist_class for dist_class in self.distributions if dist_class.var_type == var_type]

    def fit(self, series: pl.Series, var_type: str,
            unique: Optional[bool] = False) -> BaseDistribution:
        """Fit a distribution to a series.

        Search for the distribution within all available distributions in the tree.

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
        return [
            "categorical", "discrete", "continuous",
            "string", "datetime", "date", "time"
        ]

    def find_distribution(self, dist_name: str, privacy: str = "none"
                          ) -> type[BaseDistribution]:
        """Find a distribution and fit keyword arguments from a name.

        This allows us to use 'faker.city' to generate a faker instance that generates cities.

        Parameters
        ----------
        dist_name:
            Name of the distribution, such as faker.city, DiscreteUniformDistribution or normal.
        privacy:
            Type of privacy to be applied.

        Returns
        -------
        tuple[Type[BaseDistribution], dict[str, Any]]:
            A distribution and the arguments to create an instance.
        """
        for dist_class in self.distributions:
            if dist_class.is_named(dist_name) and dist_class.privacy == privacy:
                return dist_class
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
            dist_class = self.find_distribution(dist)
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
            if dist_class.implements == var_dict["distribution"]["implements"]:
                return dist_class.from_dict(var_dict["distribution"])
        raise ValueError(f"Cannot find distribution with name "
                         f"'{var_dict['distribution']['implements']}'"
                         f"and type '{var_dict['type']}'.")


class BuiltinDistributionProvider(BaseDistributionProvider):
    """Distribution tree that includes the builtin distributions."""

    @property
    def name(self):
        return "builtin"

    @property
    def version(self):
        return "1.0"

    @property
    def distributions(self) -> list[type[BaseDistribution]]:
        return [
            DiscreteUniformDistribution, PoissonDistribution, UniqueKeyDistribution,
            UniformDistribution, NormalDistribution, LogNormalDistribution,
            TruncatedNormalDistribution, ExponentialDistribution,
            MultinoulliDistribution,
            RegexDistribution, UniqueRegexDistribution, FakerDistribution,
            UniformDateDistribution,
            UniformTimeDistribution,
            UniformDateTimeDistribution,
        ]


class DistributionProviderList():  # pylint: disable=too-few-public-methods
    """List of DistributionProvider's with functionality to fit distributions.

    Arguments
    ---------
    dist_packages:
        One or more distribution packages, that are denoted either with a string ("builtin")
        , DistributionProvider (BuiltinDistributionProvider()) or DistributionProvider type
        (BuiltinDistributionProvider).
        The order in which distribution providers are included matters. If a provider implements
        the same distribution at the same privacy level, then only the first will be taken into
        account.
    """
    def __init__(
            self,
            dist_packages: Union[
                str, type[BaseDistributionProvider], BaseDistributionProvider,
                list[Union[str, type[BaseDistributionProvider], BaseDistributionProvider]]]):
        if isinstance(dist_packages, (str, type, BaseDistributionProvider)):
            dist_packages = [dist_packages]
        self.dist_packages = []
        for pkg in dist_packages:
            if isinstance(pkg, str):
                self.dist_packages.append(get_distribution_provider(pkg))
            elif isinstance(pkg, type):
                self.dist_packages.append(pkg())
            elif isinstance(pkg, BaseDistributionProvider):
                self.dist_packages.append(pkg)
            raise ValueError(f"Unknown distribution package type '{type(pkg)}'")

    def fit(self, series: pl.Series,  # pylint: disable=too-many-arguments
            var_type: str,
            dist: Optional[Union[str, BaseDistribution, type]] = None,
            privacy: BasePrivacy = NoPrivacy(),
            unique: Optional[bool] = None):
        """Fit a distribution to a column/series.

        Arguments
        ---------
        series:
            The data to fit the distributions to.
        var_type:
            The variable type of the data.
        dist:
            Distribution to fit. If not supplied or None, the AIC information
            criterion will be used to determine which distribution is the most
            suitable.
        privacy:
            Level of privacy that will be used in the fit.
        unique:
            Whether the distribution should be unique or not.
        """
        if dist is not None:
            return self._fit_distribution(series, dist, privacy)
        return self._find_best_fit(series, var_type, unique, privacy)

    def _find_best_fit(self, series: pl.Series, var_type: str,
                       unique: Optional[bool],
                       privacy: BasePrivacy) -> BaseDistribution:
        """Fit a distribution to a series.

        Search for the distribution within all available distributions in the tree.

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
        dist_list = self._get_dist_list(privacy, var_type)
        if len(dist_list) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'")
        dist_instances = [d.fit(series, **privacy.fit_kwargs) for d in dist_list]
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

    def _find_distribution(self, dist_name: str, privacy: BasePrivacy = NoPrivacy()
                           ) -> type[BaseDistribution]:
        """Find a distribution and fit keyword arguments from a name.

        This allows us to use 'faker.city' to generate a faker instance that generates cities.

        Parameters
        ----------
        dist_name:
            Name of the distribution, such as faker.city, DiscreteUniformDistribution or normal.
        privacy:
            Type of privacy to be applied.

        Returns
        -------
        tuple[Type[BaseDistribution], dict[str, Any]]:
            A distribution and the arguments to create an instance.
        """
        for dist_class in self._get_dist_list(privacy):
            if dist_class.is_named(dist_name) and dist_class.privacy == privacy.name:
                return dist_class
        raise ValueError(f"Cannot find distribution with name '{dist_name}'.")

    def _fit_distribution(self, series: pl.Series,
                          dist: Union[str, Type[BaseDistribution], BaseDistribution],
                          privacy: BasePrivacy) -> BaseDistribution:
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

        if isinstance(dist, str):
            dist_class = self._find_distribution(dist)
            dist_instance = dist_class.fit(series, **privacy.fit_kwargs)
        elif inspect.isclass(dist) and issubclass(dist, BaseDistribution):
            dist_instance = dist.fit(series, **privacy.fit_kwargs)
        if isinstance(dist, BaseDistribution):
            dist_instance = dist

        if dist_instance is None:
            raise TypeError(
                f"Distribution with type {type(dist)} is not a BaseDistribution")

        return dist_instance

    def _get_dist_list(self, privacy: BasePrivacy,
                       var_type: Optional[str] = None) -> list[type[BaseDistribution]]:
        dist_list = []
        for dist_pkg in self.dist_packages:
            if var_type is None:
                dist_list.extend(dist_pkg.distributions)
            else:
                dist_list.extend(dist_pkg.get_dist_list(var_type))

        dist_list = [dist for dist in dist_list if dist.privacy == privacy.name]
        return dist_list


def get_distribution_provider(
        provider: Union[str, type, BaseDistributionProvider] = "builtin", **kwargs
        ) -> BaseDistributionProvider:
    """Get a distribution tree.

    Parameters
    ----------
    target:
        Directive to get the distribution tree.

    Returns
    -------
    BaseDistributionProvider:
        The distribution provider that was found.
    """
    if isinstance(provider, BaseDistributionProvider):
        return provider
    if isinstance(provider, type):
        return provider()

    all_disttrees = {
        entry.name: entry
        for entry in entry_points(group="metasynth.distribution_provider")
    }
    try:
        return all_disttrees[provider].load()(**kwargs)
    except KeyError as exc:
        raise ValueError(f"Cannot find distribution provider with name '{provider}'.") from exc
