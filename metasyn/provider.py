"""Module implementing distribution providers.

Distribution providers are used to find/fit distributions that are available.
See pyproject.toml on how the builtin distribution provider is registered.
"""

from __future__ import annotations

import warnings
from abc import ABC
from inspect import signature
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

try:
    from importlib_metadata import EntryPoint, entry_points
except ImportError:
    from importlib.metadata import EntryPoint, entry_points  # type: ignore

import numpy as np
import polars as pl

from metasyn.distribution.base import BaseDistribution, BaseFitter
from metasyn.distribution.categorical import MultinoulliFitter
from metasyn.distribution.constant import (
    ContinuousConstantFitter,
    DateConstantFitter,
    DateTimeConstantFitter,
    DiscreteConstantFitter,
    StringConstantFitter,
    TimeConstantFitter,
)
from metasyn.distribution.exponential import ExponentialFitter
from metasyn.distribution.faker import FakerFitter, UniqueFakerFitter
from metasyn.distribution.freetext import FreeTextFitter
from metasyn.distribution.na import NADistribution, NAFitter
from metasyn.distribution.normal import (
    ContinuousNormalFitter,
    DiscreteNormalFitter,
    DiscreteTruncatedNormalFitter,
    LogNormalFitter,
    TruncatedNormalFitter,
)
from metasyn.distribution.poisson import PoissonFitter
from metasyn.distribution.regex import (
    RegexFitter,
    UniqueRegexFitter,
)
from metasyn.distribution.uniform import (
    ContinuousUniformFitter,
    DateTimeUniformFitter,
    DateUniformFitter,
    DiscreteUniformFitter,
    TimeUniformFitter,
)
from metasyn.distribution.uniquekey import UniqueKeyFitter
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
    accessed with ``get_fitters``.
    """

    name = ""
    version = ""
    fitters: list[type[BaseFitter]] = []

    def __init__(self):
        # Perform internal consistency check.
        assert len(self.name) > 0
        assert len(self.version) > 0
        assert len(self.fitters) > 0

    def get_fitters(self, privacy: BasePrivacy, var_type: Optional[str],
                    unique: bool = False) -> List[Type[BaseDistribution]]:
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
        fitters = [f for f in self.fitters if f.distribution.unique == unique]
        fitters = [f for f in fitters if f.privacy_type == privacy.name]
        if var_type is None:
            return fitters

        filtered_list = []
        for fit_class in fitters:
            str_chk = (isinstance(fit_class.var_type, str) and var_type == fit_class.var_type)
            lst_chk = (not isinstance(fit_class.var_type, str) and var_type in fit_class.var_type)
            if str_chk or lst_chk:
                filtered_list.append(fit_class)
        return filtered_list

    @property
    def all_var_types(self) -> List[str]:
        """Return list of available variable types."""
        var_type_set = set()
        for fitter in self.fitters:
            if isinstance(fitter.var_type, str):
                var_type_set.add(fitter.var_type)
            else:
                var_type_set.update(fitter.var_type)
        return list(var_type_set)


class BuiltinDistributionProvider(BaseDistributionProvider):
    """Distribution tree that includes the builtin distributions.

    This class inherits from BaseDistributionProvider and provides
    the built-in metasyn distributions.
    """

    name = "builtin"
    version = "1.2"
    fitters = [
        DiscreteUniformFitter, ContinuousUniformFitter, DateUniformFitter, TimeUniformFitter,
        DateTimeUniformFitter,
        RegexFitter, UniqueRegexFitter,
        ContinuousConstantFitter, DiscreteConstantFitter, DateConstantFitter,
        DateTimeConstantFitter, TimeConstantFitter, StringConstantFitter,
        ExponentialFitter,
        MultinoulliFitter,
        FakerFitter, UniqueFakerFitter,
        FreeTextFitter,
        PoissonFitter,
        ContinuousNormalFitter, LogNormalFitter, DiscreteTruncatedNormalFitter,
        TruncatedNormalFitter, DiscreteNormalFitter,
        UniqueKeyFitter,
        NAFitter,
    ]

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
        If a provider name the same distribution at the same privacy level,
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
        if dist_spec.name is not None:
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
        if dist_spec.name is None:
            raise ValueError("Cannot create distribution without specifying the 'name' key.")
        dist_class = self.find_distribution(
            dist_spec.name, var_spec.var_type, unique=unique)
        # print(dist_class)
        # print(dist_spec.parameters)
        try:
            return dist_class(**dist_spec.parameters)  # type: ignore
        except TypeError as err:
            dist_param = set(signature(dist_class.__init__).parameters) - {"self"}  # type: ignore
            unknown_param = set(dist_spec.parameters) - dist_param  # type: ignore
            missing_param = dist_param - set(dist_spec.parameters)  # type: ignore
            if len(unknown_param) > 0:
                raise TypeError(f"Unknown parameters {unknown_param} for variable {var_spec.name}."
                                f"Available parameters: {dist_param}")
            if len(missing_param) > 0:
                raise ValueError(f"Missing parameters for variable {var_spec.name}:"
                                 f" {missing_param}.")
            raise err

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
        fitters = self.get_fitters(privacy=privacy, var_type=var_type, unique=try_unique)
        if len(fitters) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'"
                             f" and unique={try_unique}")
        fit_instances = [f(privacy) for f in fitters]
        dist_instances = [d.fit(series) for d in fit_instances]
        dist_bic = [d.information_criterion(series) for d in dist_instances]
        if unique is None:
            dist_list_unq = self.get_fitters(privacy=privacy, var_type=var_type, unique=True)
            if len(dist_list_unq) > 0:
                fit_inst_unq = [f(privacy) for f in dist_list_unq]
                dist_inst_unq = [d.fit(series) for d in fit_inst_unq]
                dist_bic_unq = [d.information_criterion(series) for d in dist_inst_unq]
                # We don't want to warn about potential uniqueness too easily
                # The offset is a heuristic that ensures about 12 rows are needed for uniqueness
                # Or 5 rows for consecutive values.
                if np.min(dist_bic_unq) + 16 < np.min(dist_bic):
                    best_dist = dist_inst_unq[np.argmin(dist_bic_unq)]
                    if best_dist.name == "core.unique_key" and best_dist.consecutive:  # type: ignore
                        return best_dist
                    warnings.warn(
                        f"\nMetasyn detected that variable '{series.name}' is potentially unique.\n"
                        f"Use var_spec=[VarSpec(\"{series.name}\", unique=True)] to make it unique."
                        f"\nTo dismiss this warning use [VarSpec(\"{series.name}\", unique=False)]."
                        "\nIf you are using a configuration file add distribution = {unique = True}"
                        f" for the variable with name '{series.name}'.",
                        UserWarning
                    )

        return dist_instances[np.argmin(dist_bic)]

    def find_distribution(
            self,
            dist_name: str,
            var_type: Optional[str],
            unique: bool = False,
            version: Optional[str] = None
        ) -> type[BaseDistribution]:
        same_version = []
        for dist_class in self.get_distributions(var_type=var_type, unique=unique):
            if dist_class.matches_name(dist_name):
                if version is None or version == dist_class.version:
                    return dist_class
                same_version.append(dist_class.version)
        if len(same_version) == 0:
            raise ValueError(f"Could not find distribution with name {dist_name}.")
        raise ValueError(f"Could not find correct version of distribution with name {dist_name}."
                         f"need {version}, available: {same_version}")

    def find_fitter(self,
                          dist_name: str,
                          var_type: Optional[str],
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
        versions_found = []
        for dist_class in self.get_fitters(
                privacy=privacy, var_type=var_type, unique=unique):
            if dist_class.matches_name(dist_name):
                if version is None or version == dist_class.version:
                    return dist_class
                versions_found.append(dist_class)
        if version is None:
            raise ValueError(f"Could not find fitter with name {dist_name}.")
        raise ValueError(f"Could not find correct fitter version ({version}) of distribution with "
                         f"name {dist_name}, only found versions "
                         f"{[f.distribution.version for f in versions_found]}")

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
        assert dist_spec.name is not None

        # If the parameters are already specified, the privacy level doesn't matter anymore.
        if dist_spec.parameters is not None:
            dist_class = self.find_distribution(dist_spec.name, var_type, unique=unique)
            return dist_class(**dist_spec.parameters)

        dist_class = self.find_fitter(dist_spec.name, var_type, privacy=privacy,
                                      unique=unique)

        fit_kwargs = dist_spec.fit_kwargs
        dist_instance = dist_class(privacy).fit(series, **fit_kwargs)
        return dist_instance

    def get_fitters(self, privacy: Optional[BasePrivacy] = None,
                    var_type: Optional[str] = None,
                    unique: bool = False) -> list[type[BaseDistribution]]:
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
        fitters = []
        for dist_provider in self.dist_packages:
            fitters.extend(dist_provider.get_fitters(
                privacy=privacy, var_type=var_type, unique=unique))

        if privacy is None:
            return fitters
        fitters = [f for f in fitters if f.privacy_type == privacy.name]
        return fitters

    def get_distributions(self, var_type: Optional[str] = None,
                          unique: Optional[bool] = False):
        filtered_dist = []
        for dist in self.distributions:
            if var_type is not None:
                if isinstance(dist.var_type, str) and dist.var_type != var_type:
                    continue
                elif not isinstance(dist.var_type, str) and var_type not in dist.var_type:
                    continue
            if unique is not None and dist.unique != unique:
                continue
            filtered_dist.append(dist)
        return filtered_dist

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
        try:
            dist_name = var_dict["distribution"]["name"]
        except KeyError:
            dist_name = var_dict["distribution"]["implements"]
        version = var_dict["distribution"].get("version", "1.0")
        var_type = var_dict["type"]
        unique = var_dict["distribution"]["unique"]
        dist_class = self.find_distribution(dist_name, version=version,
                                            var_type=var_type, unique=unique)
        return dist_class.from_dict(var_dict["distribution"])

    @property
    def fitters(self):
        fitters = []
        for dist_provider in self.dist_packages:
            fitters.extend(dist_provider.fitters)
        return fitters

    @property
    def distributions(self):
        return [f.distribution for f in self.fitters]



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
