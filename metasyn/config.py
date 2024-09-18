"""Module defining configuration classes for creating MetaFrames."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from metasyn.provider import DistributionProviderList
from metasyn.varspec import DistributionSpec, VarDefaults, VarSpec


class MetaConfig():
    """Configuration class for creating MetaFrames.

    This class is used to create, manipulate, and retrieve configurations for
    individual variables in a MetaFrame. It also provides methods for loading
    configurations from .toml files and converting them to dictionaries.

    Parameters
    ----------
    var_specs:
        List of configurations for individual variables. The order does not
        matter for variables that are found in the DataFrame, but in the case
        of variables that are data-free, the order is also the order of columns
        for the eventual synthesized dataframe. See the VarSpecAccess class on
        how the dictionary can be constructed.
    dist_providers:
        Distribution providers to use when fitting distributions to variables.
        Can be a string, provider, or provider type.
    privacy:
        Privacy method/level to use as a default setting for the privacy. Can be
        overridden in the var_spec for a particular column.
    n_rows:
        Number of rows for synthesization at a later stage. Can be unspecified by
        leaving the value at None.
    """

    def __init__(
            self,
            var_specs: Union[list[dict], list[VarSpec]],
            dist_providers: Union[DistributionProviderList, list[str], str, None],
            defaults: Optional[dict] = None,
            n_rows: Optional[int] = None):
        self.var_specs = [self._parse_var_spec(v) for v in var_specs]
        self.dist_providers = dist_providers  # type: ignore
        self.n_rows = n_rows
        self.defaults = VarDefaults(**defaults)

    @staticmethod
    def _parse_var_spec(var_spec):
        if isinstance(var_spec, VarSpec):
            return var_spec
        return VarSpec.from_dict(var_spec)

    @property
    def dist_providers(self) -> DistributionProviderList:
        """Return the distribution provider list to be used for the metaframe."""
        return self._dist_providers

    @dist_providers.setter
    def dist_providers(self, dist_providers):
        if not isinstance(dist_providers, DistributionProviderList):
            self._dist_providers = DistributionProviderList(dist_providers)
        else:
            self._dist_providers = dist_providers

    @classmethod
    def from_toml(cls, config_fp: Union[str, Path]) -> MetaConfig:
        """Create a MetaConfig class from a .toml file.

        Parameters
        ----------
        config_fp:
            Path to the file containing the configuration.

        Returns
        -------
        meta_config:
            A fully initialized MetaConfig instance.
        """
        try:
            with open(config_fp, "rb") as handle:
                config_dict = tomllib.load(handle)
        except FileNotFoundError as fnf_error:
            raise FileNotFoundError(f"It appears '{config_fp}' is not a valid filepath."
                                    f" Please provide a path to a .toml file to load a MetaConfig"
                                    f" from.") from fnf_error
        except tomllib.TOMLDecodeError as value_error:
            if Path(config_fp).suffix != ".toml":
                raise ValueError(f"It appears '{Path(config_fp).name}' is a"
                                 f" '{Path(config_fp).suffix}' file."
                                 f" To load a MetaConfig, "
                                 f"provide the configuration as a .toml file.") from value_error
            raise value_error
        var_list = config_dict.pop("var", [])
        n_rows = config_dict.pop("n_rows", None)
        dist_providers = config_dict.pop("dist_providers", ["builtin"])
        defaults = config_dict.pop("defaults", None)
        privacy = config_dict.pop("privacy", None)
        config_version = config_dict.pop("config_version", "1.0")
        if config_version not in ["1.0", "1.1"]:
            warnings.warn(f"Trying to read configuration file with version {config_version}, "
                          "this version of metasyn only supports 1.0 and 1.1.")
        if privacy is not None:
            if defaults is not None:
                raise ValueError("Error parsing configuration file: cannot have both [privacy]"
                                 " and [defaults] tables.")
            defaults = {"privacy": privacy}
        if len(config_dict) > 0:
            raise ValueError(f"Error parsing configuration file '{config_fp}'."
                             f" Unknown keys detected: '{list(config_dict)}'")
        return cls(var_list, dist_providers, defaults, n_rows=n_rows)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns
        -------
        config_dict:
            Configuration in dictionary form.
        """
        return {
            "general": {
                "privacy": self.privacy,
                "dist_providers": self.dist_providers,
            },
            "var": self.var_specs
        }

    def get(self, name: str) -> VarSpecAccess:
        """Create a VarSpecAccess object pointing to a var with that name.

        If the variable does not exist, then a new variable config is created that
        has the default values.

        Parameters
        ----------
        name:
            Name of the variable configuration to retrieve.

        Returns
        -------
        var_spec:
            A variable config access object.
        """
        for var_spec in self.var_specs:
            if var_spec.name == name:
                return VarSpecAccess(var_spec, self)
        return VarSpecAccess(VarSpec(name=name), self)

    def iter_var(self, exclude: Optional[list[str]] = None) -> Iterable[VarSpecAccess]:
        """Iterate over all variables in the configuration.

        Parameters
        ----------
        exclude:
            Exclude variables with names in that list.

        Returns
        -------
        var_spec:
            VarSpecAccess class for each of the available variable configurations.
        """
        exclude = exclude if exclude is not None else []
        for var_spec in self.var_specs:
            if var_spec.name not in exclude:
                yield VarSpecAccess(var_spec, self)


class VarSpecAccess():
    """Access for variable configuration object.

    They take into account what the defaults are from the MetaConfig object.
    Otherwise they pass through all the attributes as normal and thus behave
    exactly as a variable config object themselves.

    Parameters
    ----------
    var_spec
        The variable configuration to access.
    meta_config
        The meta configuration instance to get default values from.
    """

    def __init__(self, var_spec: VarSpec, meta_config: MetaConfig):
        self.var_spec = var_spec
        self.meta_config = meta_config

    def __getattribute__(self, attr):
        if attr in ["privacy", "data_free", "prop_missing", "privacy"]:
            if getattr(self.var_spec, attr) is None:
                return getattr(self.meta_config.defaults, attr)
            return getattr(self.var_spec, attr)
        if attr == "dist_spec":
            if self.data_free and self.var_spec.dist_spec.implements is None:
                if self.var_type not in self.meta_config.defaults.distribution:
                    raise ValueError(
                        f"Variable with name '{self.name}' is declared datafree and a distribution "
                        f"cannot be inferred. You can set the distribution for '{self.name}' or set"
                        f" a default distribution for variable type '{self.var_type}'.")
                return DistributionSpec.parse(self.meta_config.defaults.distribution[self.var_type])
            return self.var_spec.dist_spec

        if attr not in ("var_spec", "meta_config") and hasattr(self.var_spec, attr):
            return getattr(self.var_spec, attr)
        return super().__getattribute__(attr)
