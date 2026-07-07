"""Module implementing the distribution registry.

Distribution registries are used to find/fit distributions that are available.
See pyproject.toml on how the builtin distributions are registered.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Any, Optional

from metasyn.distribution.base import BaseDistribution, BaseFitter
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.util import get_registry


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
    def parse(cls, plugins: list[str] | None | str | DistributionRegistry):
        """Initialize the distribution registry from plugin names.

        Parameters
        ----------
        plugins:
            Name of plugin(s) for fitters/distribution or a list of names.
        """
        if isinstance(plugins, DistributionRegistry):
            return plugins
        fitters = []
        if isinstance(plugins, str):
            plugins = [plugins]

        entries = {e.name: e for e in entry_points(group="metasyn.distribution_registry")}
        if plugins is None:
            plugins = list(entries)

        for registry_name in plugins:
            if registry_name not in entries:
                registry = get_registry()
                if registry_name not in registry:
                    raise ValueError(
                        f"Cannot find plugin with name '{registry_name}'.")
                raise ValueError(
                    f"Plugin '{registry_name}' is not installed.\n"
                    f"See {registry[registry_name]['url']} for installation instructions."
                )
            try:
                fitters.extend(entries[registry_name].load())
            except Exception as exc:
                warnings.warn(f"Could not load plugin with name {registry_name}, plugin might be"
                              f" broken or out of date: {exc}")
        return cls(fitters)

    def find_distribution(
            self,
            name: str,
            var_type: str | None,
            unique: bool = False,
            version: str | None = None
        ) -> type[BaseDistribution]:
        dist_classes = self.filter_distributions(name=name, var_type=var_type,
                                                 unique=unique, version=version)
        if len(dist_classes) == 1:
            return dist_classes[0]

        if len(dist_classes) > 1:
            dist_str = [f"({d.__name__}, {d.var_type}, {d.unique}, {d.version})"
                        for d in dist_classes]
            raise ValueError(f"Multiple valid distributions found with name {name}, var_type "
                             f"{var_type}, unique {unique}, version {version}."
                             f" Alternatives: {dist_str}")
        name_classes = self.filter_distributions(name=name)
        if len(name_classes) == 0:
            raise ValueError(f"No known distributions with name '{name}'.")
        dist_str = [f"({d.__name__}, {d.var_type}, {d.unique}, {d.version})"
            for d in name_classes]
        raise ValueError(f"No distribution found with name {name}, var_type "
                         f"{var_type}, unique {unique}, version {version}."
                         f" Alternatives: {dist_str}")

    def find_fitters(self,
                     dist_name: str,
                     var_type: Optional[str],
                     privacy: Optional[BasePrivacy] = BasicPrivacy(),
                     unique: bool = False,
                     version: Optional[str] = None) -> list[type[BaseFitter]]:
        """Find a distribution and fit keyword arguments from a name.

        Sometimes there might be multiple possible fitters that satisfy the criteria.
        In this case the first in the registry will be chosen. If you do not want this
        behavior, it is recommended to specify the fitter name directly.

        Parameters
        ----------
        dist_name:
            Name of the distribution that needs to be fit, e.g., for the built-in
            uniform distribution: "uniform", "core.uniform"
            or name of the fitter: "ContinuousUniformFitter".
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
        tuple[Type[BaseFitter]:
            Fitter that satisfies the requirements.
        """
        fitter_classes = self.filter_fitters(
            name=dist_name, privacy=privacy, var_type=var_type, unique=unique, version=version)
        if len(fitter_classes) == 1:
            return fitter_classes

        if len(fitter_classes) > 1:
            if var_type is None and not all([
                    f.var_type == fitter_classes[0] for f in fitter_classes]):
                raise ValueError(f"Multiple valid fitters found with name {dist_name}, "
                                 "please specify var_type.")
            return fitter_classes

        name_classes = self.filter_fitters(name=dist_name)
        if len(name_classes) == 0:
            raise ValueError(f"No known fitters with name '{dist_name}'.")
        fitter_str = [f"({f.__name__}, {f.var_type}, {f.distribution.unique}, {f.version},"
                      f" {f.privacy_type})" for f in name_classes]
        raise ValueError(f"No fitter found with name {dist_name}, var_type "
                         f"{var_type}, unique {unique}, version {version}."
                         f" Alternatives: {fitter_str}")

    def filter_fitters(self,
                       name: Optional[str] = None,
                       privacy: Optional[BasePrivacy] = None,
                       var_type: Optional[str] = None,
                       unique: bool = False,
                       version: Optional[str] = None) -> list[type[BaseFitter]]:
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
        if name is not None:
            fitters = [f for f in fitters if f.matches_name(name)]
        if var_type is not None:
            fitters = [f for f in fitters if f.provides_var_type(var_type)]
        fitters = [f for f in fitters if f.distribution.unique == unique]
        if privacy is not None:
            fitters = [f for f in fitters if f.privacy_type == privacy.name]
        if version is not None:
            fitters = [f for f in fitters if f.version == version]
        return fitters

    def filter_distributions(self, name: Optional[str] = None, var_type: Optional[str] = None,
                             unique: Optional[bool] = False, version: Optional[str] = None):
        dist = self.distributions

        if name is not None:
            dist = [d for d in dist if d.matches_name(name)]

        if var_type is not None:
            dist = [d for d in dist if d.provides_var_type(var_type)]

        if unique is not None:
            dist = [d for d in dist if d.unique == unique]

        if version is not None:
            dist = [d for d in dist if d.version == version]

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
        """All available distributions from fitters, deduplicated."""
        dists = [f.distribution for f in self.fitters]
        # Deduplicate distributions
        new_dists = []
        dist_names = set()
        for dist in dists:
            if dist not in dist_names:
                new_dists.append(dist)
                dist_names.add(dist)
        return new_dists
