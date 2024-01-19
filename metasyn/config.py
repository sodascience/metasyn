"""Module defining configuration classes for creating MetaFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from metasyn.privacy import BasePrivacy, get_privacy
from metasyn.provider import DistributionProviderList
from metasyn.util import VarConfig


class MetaConfig():
    """Configuration class for creating MetaFrames.

    This class is used to create, manipulate, and retrieve configurations for
    individual variables in a MetaFrame. It also provides methods for loading
    configurations from .toml files and converting them to dictionaries.

    Parameters
    ----------
    var_configs:
        List of configurations for individual variables. The order does not
        matter for variables that are found in the DataFrame, but in the case
        of variables that are data-free, the order is also the order of columns
        for the eventual synthesized dataframe. See the VarConfigAccess class on
        how the dictionary can be constructed.
    dist_providers:
        Distribution providers to use when fitting distributions to variables.
        Can be a string, provider, or provider type.
    privacy:
        Privacy method/level to use as a default setting for the privacy. Can be
        overridden in the var_config for a particular column.
    n_rows:
        Number of rows for synthesization at a later stage. Can be unspecified by
        leaving the value at None.
    """

    def __init__(
            self,
            var_configs: Union[list[dict], list[VarConfig]],
            dist_providers: Union[DistributionProviderList, list[str], str],
            privacy: Union[BasePrivacy, dict],
            n_rows: Optional[int] = None):
        self.var_configs = [self._parse_var_config(v) for v in var_configs]

        if not isinstance(dist_providers, DistributionProviderList):
            dist_providers = DistributionProviderList(dist_providers)
        self.dist_providers = dist_providers

        if not isinstance(privacy, BasePrivacy):
            privacy = get_privacy(**privacy)

        self.privacy = privacy
        self.n_rows = n_rows

    @staticmethod
    def _parse_var_config(var_cfg):
        if isinstance(var_cfg, VarConfig):
            return var_cfg
        return VarConfig.from_dict(var_cfg)

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
        with open(config_fp, "rb") as handle:
            config_dict = tomllib.load(handle)
        general = config_dict.get("general", {})
        var_list = config_dict.pop("var", [])
        n_rows = general.pop("n_rows", None)
        dist_providers = general.pop("dist_providers", ["builtin"])
        privacy = general.pop("privacy", {"name": "none", "parameters": {}})
        if len(general) > 0:
            raise ValueError(f"Error parsing configuration file '{config_fp}'."
                             f" Unknown keys detected: '{list(general)}'")
        return cls(var_list, dist_providers, privacy, n_rows=n_rows)

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
            "var": self.var_configs
        }

    def get(self, name: str) -> VarConfigAccess:
        """Create a VarConfigAccess object pointing to a var with that name.

        If the variable does not exist, then a new variable config is created that
        has the default values.

        Parameters
        ----------
        name:
            Name of the variable configuration to retrieve.

        Returns
        -------
        var_cfg:
            A variable config access object.
        """
        for var_cfg in self.var_configs:
            if var_cfg.name == name:
                return VarConfigAccess(var_cfg, self)
        return VarConfigAccess(VarConfig(name=name), self)

    def iter_var(self, exclude: Optional[list[str]] = None) -> Iterable[VarConfigAccess]:
        """Iterate over all variables in the configuration.

        Parameters
        ----------
        exclude:
            Exclude variables with names in that list.

        Returns
        -------
        var_cfg:
            VarConfigAccess class for each of the available variable configurations.
        """
        exclude = exclude if exclude is not None else []
        for var_spec in self.var_configs:
            if var_spec.name not in exclude:
                yield VarConfigAccess(var_spec, self)


class VarConfigAccess():  # pylint: disable=too-few-public-methods
    """Access for variable configuration object.

    They take into account what the defaults are from the MetaConfig object.
    Otherwise they pass through all the attributes as normal and thus behave
    exactly as a variable config object themselves.

    Parameters
    ----------
    var_config
        The variable configuration to access.
    meta_config
        The meta configuration instance to get default values from.
    """

    def __init__(self, var_config: VarConfig, meta_config: MetaConfig):
        self.var_config = var_config
        self.meta_config = meta_config

    def __getattribute__(self, attr):
        if attr == "privacy":
            if self.var_config.privacy is None:
                return self.meta_config.privacy
            return self.var_config.privacy
        if attr not in ("var_config", "meta_config") and hasattr(self.var_config, attr):
            return getattr(self.var_config, attr)
        return super().__getattribute__(attr)
