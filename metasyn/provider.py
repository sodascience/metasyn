"""Module implementing distribution providers.

Distribution providers are used to find/fit distributions that are available.
See pyproject.toml on how the builtin distribution provider is registered.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

try:
    from importlib_metadata import EntryPoint, entry_points
except ImportError:
    from importlib.metadata import EntryPoint, entry_points  # type: ignore

import numpy as np
import polars as pl

from metasyn.distribution.base import BaseDistribution
from metasyn.distribution.categorical import MultinoulliDistribution
from metasyn.distribution.continuous import (
    ConstantDistribution,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)
from metasyn.distribution.datetime import (
    DateConstantDistribution,
    DateTimeConstantDistribution,
    DateTimeUniformDistribution,
    DateUniformDistribution,
    TimeConstantDistribution,
    TimeUniformDistribution,
)
from metasyn.distribution.discrete import (
    DiscreteConstantDistribution,
    DiscreteNormalDistribution,
    DiscreteTruncatedNormalDistribution,
    DiscreteUniformDistribution,
    PoissonDistribution,
    UniqueKeyDistribution,
)
from metasyn.distribution.na import NADistribution
from metasyn.distribution.string import (
    FakerDistribution,
    FreeTextDistribution,
    RegexDistribution,
    StringConstantDistribution,
    UniqueFakerDistribution,
    UniqueRegexDistribution,
)
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.util import get_registry
from metasyn.varspec import DistributionSpec

if TYPE_CHECKING:
    from metasyn.config import VarSpec, VarSpecAccess


class BaseDistributionProvider(ABC):
    """Base class for all distribution providers.

    A distribution provider is a class that provides a set of distributions
    that can be used by metasyn to generate synthetic data.
    This class acts as a base class for creating specific distribution
    providers. It also contains a list of the available distributions and
    legacy distributions. A list of distributions for a specific type can be
    accessed with ``get_dist_list``.
    """

    name = ""
    version = ""
    distributions: list[type[BaseDistribution]] = []
    legacy_distributions: list[type[BaseDistribution]] = []

    def __init__(self):
        # Perform internal consistency check.
        assert len(self.name) > 0
        assert len(self.version) > 0
        assert len(self.distributions) > 0

    def get_dist_list(self, var_type: Optional[str],
                      unique: bool = False,
                      use_legacy: bool = False) -> List[Type[BaseDistribution]]:
        """Get all distributions for a certain variable type.

        Parameters
        ----------
        var_type:
            Variable type to get the distributions for.
        unique:
            Whether the distirbutions should be unique.
        use_legacy:
            Whether to find the distributions in the legacy distribution list.

        Returns
        -------
        list[Type[BaseDistribution]]:
            List of distributions with that variable type.
        """
        dist_list = []
        if use_legacy:
            distributions = self.legacy_distributions
        else:
            distributions = self.distributions
        distributions = [d for d in distributions if d.unique == unique]
        if var_type is None:
            return distributions
        for dist_class in distributions:
            str_chk = (isinstance(dist_class.var_type, str) and var_type == dist_class.var_type)
            lst_chk = (not isinstance(dist_class.var_type, str) and var_type in dist_class.var_type)
            if str_chk or lst_chk:
                dist_list.append(dist_class)
        return dist_list

    @property
    def all_var_types(self) -> List[str]:
        """Return list of available variable types."""
        var_type_set = set()
        for dist in self.distributions:
            if isinstance(dist.var_type, str):
                var_type_set.add(dist.var_type)
            else:
                var_type_set.update(dist.var_type)
        return list(var_type_set)


class BuiltinDistributionProvider(BaseDistributionProvider):
    """Distribution tree that includes the builtin distributions.

    This class inherits from BaseDistributionProvider and provides
    the built-in metasyn distributions.
    """

    name = "builtin"
    version = "1.2"
    distributions = [
        DiscreteNormalDistribution, DiscreteTruncatedNormalDistribution,
        DiscreteUniformDistribution, PoissonDistribution, UniqueKeyDistribution,
        UniformDistribution, NormalDistribution, LogNormalDistribution,
        TruncatedNormalDistribution, ExponentialDistribution,
        MultinoulliDistribution,
        RegexDistribution, UniqueRegexDistribution, FakerDistribution, UniqueFakerDistribution,
        FreeTextDistribution,
        DateUniformDistribution,
        TimeUniformDistribution,
        DateTimeUniformDistribution,
        ConstantDistribution,
        DiscreteConstantDistribution,
        StringConstantDistribution,
        DateTimeConstantDistribution,
        DateConstantDistribution,
        TimeConstantDistribution,
    ]
    legacy_distributions = []


class DistributionProviderList():
    """List of DistributionProviders with functionality to fit distributions.

    This class is responsible for managing and providing access to
    different distribution providers. It allows for fitting distributions,
    as well as retrieving distributions based on certain constraints
    such as privacy level, variable type, and uniqueness.

    Parameters
    ----------
    dist_providers:
        One or more distribution providers, that are denoted either with a string ("builtin"),
        DistributionProvider (BuiltinDistributionProvider())
        or DistributionProvider type (BuiltinDistributionProvider).
        The order in which distribution providers are included matters.
        If a provider implements the same distribution at the same privacy level,
        then only the first will be taken into account.
    """

    def __init__(
            self,
            dist_providers: Union[
                list[str],
                None, str, type[BaseDistributionProvider], BaseDistributionProvider,
                list[Union[str, type[BaseDistributionProvider], BaseDistributionProvider]]]):
        if dist_providers is None:
            self.dist_packages = _get_all_provider_list()
            return

        if isinstance(dist_providers, (str, type, BaseDistributionProvider)):
            dist_providers = [dist_providers]
        self.dist_packages = []
        for provider in dist_providers:
            if isinstance(provider, str):
                self.dist_packages.append(get_distribution_provider(provider))
            elif isinstance(provider, type):
                self.dist_packages.append(provider())
            elif isinstance(provider, BaseDistributionProvider):
                self.dist_packages.append(provider)
            else:
                raise ValueError(f"Unknown distribution package type '{type(provider)}'")

    def fit(self, series: pl.Series,
            var_type: str,
            dist_spec: DistributionSpec,
            privacy: BasePrivacy = BasicPrivacy()) -> BaseDistribution:
        """Fit a distribution to a column/series.

        Parameters
        ----------
        series:
            The data to fit the distributions to.
        var_type:
            The variable type of the data.
        dist_spec:
            Distribution to fit. If not supplied or None, the information
            criterion will be used to determine which distribution is the most
            suitable. For most variable types, the information criterion is based on
            the BIC (Bayesian Information Criterion).
        privacy:
            Level of privacy that will be used in the fit.
        """
        if dist_spec.distribution is not None:
            return dist_spec.distribution
        if dist_spec.implements is not None:
            return self._fit_distribution(series, dist_spec, var_type, privacy)
        return self._find_best_fit(series, var_type, dist_spec.unique, privacy)

    def create(self, var_spec: Union[VarSpec, VarSpecAccess]) -> BaseDistribution:
        """Create a distribution without any data.

        Parameters
        ----------
        var_spec
            A variable configuration that provides all the qinformation to create the distribution.

        Returns
        -------
            A distribution according to the variable specifications.
        """
        dist_spec = var_spec.dist_spec
        unique = dist_spec.unique if dist_spec.unique else False
        assert dist_spec.implements is not None and var_spec.var_type is not None
        dist_class = self.find_distribution(
            dist_spec.implements, var_spec.var_type,
            privacy=BasicPrivacy(), unique=unique)
        return dist_class(**dist_spec.parameters)

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
        privacy:
            Privacy level to find the best fit with.

        Returns
        -------
        BaseDistribution:
            Distribution fitted to the series.
        """
        if len(series.drop_nulls()) == 0:
            return NADistribution()
        try_unique = unique if unique is True else False
        dist_list = self.get_distributions(privacy, var_type, unique=try_unique)
        if len(dist_list) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'"
                             f" and unique={try_unique}")
        dist_instances = [d.fit(series, **privacy.fit_kwargs) for d in dist_list]
        dist_bic = [d.information_criterion(series) for d in dist_instances]
        if unique is None:
            dist_list_unq = self.get_distributions(privacy, var_type, unique=True)
            if len(dist_list_unq) > 0:
                dist_inst_unq = [d.fit(series, **privacy.fit_kwargs) for d in dist_list_unq]
                dist_bic_unq = [d.information_criterion(series) for d in dist_inst_unq]
                if np.min(dist_bic_unq) < np.min(dist_bic):
                    warnings.warn(
                        f"\nVariable '{series.name}' was detected to be unique, but has not"
                        f" explicitly been set to unique.\n"
                        f"To generate only unique values for column '{series.name}', "
                        f"set unique to True.\n"
                        f"To dismiss this warning, set unique to False.",
                        UserWarning
                    )

        return dist_instances[np.argmin(dist_bic)]

    def find_distribution(self,  # pylint: disable=too-many-branches
                          dist_name: str,
                          var_type: str,
                          privacy: Optional[BasePrivacy] = BasicPrivacy(),
                          unique: bool = False,
                          version: Optional[str] = None) -> type[BaseDistribution]:
        """Find a distribution and fit keyword arguments from a name.

        Parameters
        ----------
        dist_name:
            Name of the distribution, e.g., for the built-in
            uniform distribution: "uniform", "core.uniform", "UniformDistribution".
        privacy:
            Type of privacy to be applied.
        var_type:
            Type of the variable to find. If var_type is None, then do not check the
            variable type.
        unique:
            Whether the distribution to be found is unique.
        version:
            Version of the distribution to get. If necessary get them from legacy.

        Returns
        -------
        tuple[Type[BaseDistribution], dict[str, Any]]:
            A distribution and the arguments to create an instance.
        """
        if NADistribution.matches_name(dist_name):
            return NADistribution

        versions_found = []
        for dist_class in self.get_distributions(
                privacy, var_type=var_type, unique=unique) + [NADistribution]:
            if dist_class.matches_name(dist_name):
                if version is None or version == dist_class.version:
                    return dist_class
                versions_found.append(dist_class)

        # Look for distribution in legacy
        warnings.simplefilter("always")
        legacy_distribs = [
            dist_class
            for dist_class in self.get_distributions(privacy, use_legacy=True,
                                                     var_type=var_type, unique=unique)
            if dist_class.matches_name(dist_name)]

        if len(legacy_distribs + versions_found) == 0:
            registry = get_registry()
            available = {}
            for plugin, meta in registry.items():
                if dist_name in meta["distributions"]:
                    available[plugin] = meta["url"]
            if len(available) > 0:
                avail_str = "\n".join("{plugin}: {url}" for plugin, url in available.items())
                raise ValueError(f"You are trying to use a distribution named '{dist_name}', \n"
                                 f"but it is not installed.\n"
                                 f"\n"
                                 f"{dist_name} is available from:\n\n{avail_str}\n")
            raise ValueError(f"Cannot find distribution with name '{dist_name}'.")

        if len(versions_found) == 0:
            warnings.warn("Distribution with name '{dist_name}' is deprecated and "
                          "will be removed in the future.")

        # Find exact matches in legacy distributions
        legacy_versions = [dist.version for dist in legacy_distribs]
        if version is not None and version in legacy_versions:
            warnings.warn("Version ({version}) of distribution with name '{dist_name}'"
                          " is deprecated and will be removed in the future.")
            return legacy_distribs[legacy_versions.index(version)]

        # If version is None, take the latest version.
        if version is None:
            scores = [int(dist.version.split(".")[0]) * 100 + int(dist.version.split(".")[1])
                      for dist in legacy_distribs]
            i_min = np.argmax(scores)
            return legacy_distribs[i_min]

        # Find the distribution with the same major revision, and closest minor revision.
        major_version = int(version.split(".")[0])
        minor_version = int(version.split(".")[1])
        all_dist = versions_found + legacy_distribs
        all_versions = [[int(x) for x in dist.version.split(".")] for dist in all_dist]
        score = [int(ver[0] == major_version) * 1000000 - (ver[1] - minor_version) ** 2
                 for ver in all_versions]
        i_max = np.argmax(score)

        # Wrong major revision.
        if score[i_max] < 500000:
            raise ValueError(
                f"Cannot find compatible version for distribution '{dist_name}', available: "
                f"{legacy_versions + versions_found}")

        warnings.warn("Version mismatch ({version}) versus ({all_dist[i_max].version}))")
        return all_dist[i_max]

    def _fit_distribution(self, series: pl.Series,
                          dist_spec: DistributionSpec,
                          var_type: str,
                          privacy: BasePrivacy) -> BaseDistribution:
        """Fit a specific distribution to a series.

        In contrast the fit method, this needs a supplied distribution(type).

        Parameters
        ----------
        series:
            Series to fit the distribution to.
        dist_spec:
            Distribution to fit (if it is not already fitted).
        var_type:
            Type of variable to fit the distribution for.
        privacy:
            Privacy level to fit the distribution with.

        Returns
        -------
        BaseDistribution:
            Fitted distribution.
        """
        unique = dist_spec.unique
        unique = unique if unique else False
        assert dist_spec.implements is not None

        # If the parameters are already specified, the privacy level doesn't matter anymore.
        if dist_spec.parameters is not None:
            dist_class = self.find_distribution(dist_spec.implements, var_type, unique=unique)
            return dist_class(**dist_spec.parameters)

        dist_class = self.find_distribution(dist_spec.implements, var_type, privacy=privacy,
                                            unique=unique)

        if issubclass(dist_class, NADistribution):
            dist_instance = dist_class.default_distribution()
        else:
            fit_kwargs = dist_spec.fit_kwargs
            dist_instance = dist_class.fit(series, **privacy.fit_kwargs, **fit_kwargs)
        return dist_instance

    def get_distributions(self, privacy: Optional[BasePrivacy] = None,
                          var_type: Optional[str] = None,
                          unique: bool = False,
                          use_legacy: bool = False) -> list[type[BaseDistribution]]:
        """Get the available distributions with constraints.

        Parameters
        ----------
        privacy:
            Privacy level/type to filter the distributions.
        var_type:
            Variable type to filter for, e.g. 'string'.
        unique:
            Whether the distributions to be gotten are unique.
        use_legacy:
            Whether to use legacy distributions or not.

        Returns
        -------
        dist_list:
            List of distributions that fit the given constraints.
        """
        dist_list = []
        for dist_provider in self.dist_packages:
            dist_list.extend(dist_provider.get_dist_list(
                var_type, use_legacy=use_legacy, unique=unique))

        if privacy is None:
            return dist_list
        dist_list = [dist for dist in dist_list if dist.privacy == privacy.name]
        return dist_list

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
        dist_name = var_dict["distribution"]["implements"]
        version = var_dict["distribution"].get("version", "1.0")
        var_type = var_dict["type"]
        unique = var_dict["distribution"]["unique"]
        dist_class = self.find_distribution(dist_name, version=version,
                                            var_type=var_type, unique=unique)
        return dist_class.from_dict(var_dict["distribution"])


def _get_all_providers() -> dict[str, EntryPoint]:
    """Get all available providers."""
    return {
        entry.name: entry
        for entry in entry_points(group="metasyn.distribution_provider")
    }


def _get_all_provider_list() -> list[BaseDistributionProvider]:
    return [p.load()() for p in _get_all_providers().values()]


def get_distribution_provider(provider: Union[str, type[
                                        BaseDistributionProvider],
                                        BaseDistributionProvider] = "builtin"
                              ) -> BaseDistributionProvider:
    """Get a distribution tree.

    Parameters
    ----------
    provider:
        Name, class or class type of the provider to be used.
    kwargs:
        Extra keyword arguments for initialization of the distribution provider.

    Returns
    -------
    BaseDistributionProvider:
        The distribution provider that was found.
    """
    if isinstance(provider, BaseDistributionProvider):
        return provider
    if isinstance(provider, type):
        return provider()

    all_providers = _get_all_providers()
    try:
        return all_providers[provider].load()()
    except KeyError as exc:
        registry = get_registry()
        if provider not in registry:
            raise ValueError(f"Cannot find distribution provider with name '{provider}'.") from exc
        raise ValueError(f"Distribution provider '{provider}' is not installed.\n"
                         f"See {registry['provider']['url']} for installation instructions."
                         ) from exc
