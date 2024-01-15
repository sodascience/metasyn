from typing import Optional, Union

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa  

from metasyn.privacy import BasePrivacy, get_privacy
from metasyn.provider import DistributionProviderList


class MetaConfig():
    def __init__(
            self,
            var_configs: list[dict],
            dist_providers: Union[DistributionProviderList, list[str], str],
            privacy: Union[BasePrivacy, dict],
            n_rows: Optional[int] = None):
        self.var_configs = [self._parse_var_config(v) for v in var_configs]
        if isinstance(dist_providers, DistributionProviderList):
            self.dist_providers = dist_providers
        else:
            self.dist_providers = DistributionProviderList(dist_providers)
        if isinstance(privacy, BasePrivacy):
            self.privacy = privacy
        else:
            self.privacy = get_privacy(**privacy)
        self.n_rows = n_rows

    @staticmethod
    def _parse_var_config(var_cfg):
        if "privacy" in var_cfg:
            if not isinstance(var_cfg["privacy"], BasePrivacy):
                var_cfg["privacy"] = get_privacy(**var_cfg["privacy"])
        return var_cfg

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

    def __getitem__(self, name):
        return VarConfigAccess(name, self)

    def get(self, name):
        for var_dict in self.var_configs:
            if var_dict["name"] == name:
                return var_dict
        return {}

    def iter_var(self, exclude: Optional[list[str]] = None):
        exclude = exclude if exclude is not None else []
        for var_spec in self.var_configs:
            if var_spec["name"] not in exclude:
                yield self[var_spec["name"]]


class VarConfigAccess():
    def __init__(self, name:str, meta_config: MetaConfig):
        self.name = name
        self.meta_config = meta_config

    @property
    def spec(self) -> dict:
        return self.meta_config.get(self.name)

    @property
    def dist_spec(self) -> dict:
        return self.spec.get("distribution", {})

    @property
    def privacy(self) -> BasePrivacy:
        if "privacy" in self.spec:
            return self.spec["privacy"]
        return self.meta_config.privacy

    @property
    def prop_missing(self) -> Optional[float]:
        return self.spec.get("prop_missing", None)

    @property
    def description(self) -> Optional[str]:
        return self.spec.get("description", None)

    @property
    def data_free(self) -> bool:
        return self.spec.get("data_free", False)

    @property
    def var_type(self) -> str:
        return self.spec.get("var_type", None)
