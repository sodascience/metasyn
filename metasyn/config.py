"Configuration classes for creating metaframes."
from dataclasses import dataclass, field
from typing import Optional, Union

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from metasyn.privacy import BasePrivacy, get_privacy
from metasyn.provider import DistributionProviderList
from metasyn.util import VarConfig


class MetaConfig():
    """Configuration class for creating MetaFrames.

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
    def from_toml(cls, config_fp):
        with open(config_fp, "rb") as handle:
            config_dict = tomllib.load(handle)
        general = config_dict.get("general", {})
        var_dict = config_dict.pop("var", {})
        n_rows = general.pop("n_rows", None)
        dist_providers = general.pop("dist_providers", ["builtin"])
        privacy = general.pop("privacy", {"name": "none", "parameters": {}})
        if len(general) > 0:
            raise ValueError(f"Error parsing configuration file '{config_fp}'."
                             f" Unknown keys detected: '{list(general)}'")
        return cls(var_dict, dist_providers, privacy, n_rows=n_rows)

    def to_dict(self):
        return {
            "general": {
                "privacy": self.privacy,
                "dist_providers": self.dist_providers,
            },
            "var": self.var
        }

    def get(self, name):
        for var_cfg in self.var_configs:
            if var_cfg.name == name:
                return VarConfigAccess(var_cfg, self)
        return VarConfigAccess(VarConfig(name=name), self)

    def iter_var(self, exclude: Optional[list[str]] = None):
        exclude = exclude if exclude is not None else []
        for var_spec in self.var_configs:
            if var_spec.name not in exclude:
                yield VarConfigAccess(var_spec, self)


class VarConfigAccess():
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

    # @property
    # def spec(self) -> dict:
    #     return self.meta_config.get(self.name)

    # @property
    # def dist_spec(self) -> dict:
    #     return self.spec.get("distribution", {})

    # @property
    # def privacy(self) -> BasePrivacy:
    #     if "privacy" in self.spec:
    #         return self.spec["privacy"]
    #     return self.meta_config.privacy

    # @property
    # def prop_missing(self) -> Optional[float]:
    #     return self.spec.get("prop_missing", None)

    # @property
    # def description(self) -> Optional[str]:
    #     return self.spec.get("description", None)

    # @property
    # def data_free(self) -> bool:
    #     return self.spec.get("data_free", False)

    # @property
    # def var_type(self) -> str:
    #     return self.spec.get("var_type", None)
