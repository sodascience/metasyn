"""Module implementing the distribution registry.

Distribution registries are used to find/fit distributions that are available.
See pyproject.toml on how the builtin distributions are registered.
"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import TYPE_CHECKING, Any, Optional, Union

try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore

import numpy as np
import polars as pl

from metasyn.distribution.base import BaseDistribution, BaseFitter
from metasyn.distribution.na import NADistribution
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.util import get_registry
from metasyn.varspec import DistributionSpec

if TYPE_CHECKING:
    from metasyn.config import VarSpec, VarSpecAccess


class DistributionRegistry():
    """Registry of distributions and fitters.

    This class is responsible for managing and providing access to
    fitters and distributions. It allows for fitting distributions,
    as well as retrieving distributions/fitters based on certain constraints
    such as privacy level, variable type, and uniqueness.

    You can directly initialize the class with a list of fitters, but most likely
    you will want to use the :meth:`DistributionRegistry.parse` method, which can load
    fitters from registries provided by plugins.

    Parameters
    ----------
    fitters:
        Fitters to initialize the registry with.
    """

    def __init__(
            self,
            fitters: list[type[BaseFitter]]):
        self.fitters = fitters

    @classmethod
    def parse(cls, dist_registries: Union[list[str], None, str]):
        """Initialize the distribution registry from registry names.

        Parameters
        ----------
        dist_registries:
            Name of registry for fitters/distribution or a list of names.
        """
        fitters = []
        if isinstance(dist_registries, str):
            dist_registries = [dist_registries]

        entries = {e.name: e for e in entry_points(group="metasyn.distribution_registry")}
        if dist_registries is None:
            dist_registries = list(entries)

        for registry_name in dist_registries:
            if registry_name not in entries:
                registry = get_registry()
                if registry_name not in registry:
                    raise ValueError(
                        f"Cannot find distribution registry with name '{registry_name}'.")
                raise ValueError(f"Distribution registry '{registry_name}' is not installed.\n"
                                f"See {registry['registry']['url']} for installation instructions."
                                )
            fitters.extend(entries[registry_name].load())
        return cls(fitters)

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
            A variable configuration that provides all the information to create the distribution.

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
        try_unique = unique is True
        fitters = self.filter_fitters(privacy=privacy, var_type=var_type, unique=try_unique)
        if len(fitters) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'"
                             f" and unique={try_unique}")
        fit_instances = [f(privacy) for f in fitters]
        dist_instances = [d.fit(series) for d in fit_instances]
        dist_bic = [d.information_criterion(series) for d in dist_instances]
        if unique is None:
            dist_list_unq = self.filter_fitters(privacy=privacy, var_type=var_type, unique=True)
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
        for dist_class in self.filter_distributions(var_type=var_type, unique=unique):
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
                          version: Optional[str] = None) -> type[BaseFitter]:
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
        for fitter_class in self.filter_fitters(
                privacy=privacy, var_type=var_type, unique=unique):
            if fitter_class.matches_name(dist_name):
                if version is None or version == fitter_class.version:
                    return fitter_class
                versions_found.append(fitter_class)
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

        fitter_class = self.find_fitter(dist_spec.name, var_type, privacy=privacy,
                                      unique=unique)

        fit_kwargs = dist_spec.fit_kwargs
        dist_instance = fitter_class(privacy).fit(series, **fit_kwargs)
        return dist_instance

    def filter_fitters(self, privacy: Optional[BasePrivacy] = None,
                       var_type: Optional[str] = None,
                       unique: bool = False) -> list[type[BaseFitter]]:
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
        fitters = self.fitters
        if var_type is not None:
            fitters = [f for f in fitters if f.provides_var_type(var_type)]
        fitters = [f for f in fitters if f.distribution.unique == unique]
        if privacy is not None:
            fitters = [f for f in fitters if f.privacy_type == privacy.name]
        return fitters

    def filter_distributions(self, var_type: Optional[str] = None,
                          unique: Optional[bool] = False):
        dist = self.distributions

        if var_type is not None:
            dist = [d for d in dist if d.provides_var_type(var_type)]

        if unique is not None:
            dist = [d for d in dist if d.unique == unique]
        return dist

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
    def distributions(self):
        return [f.distribution for f in self.fitters]
