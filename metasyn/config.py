from typing import Optional

import polars as pl
from tqdm import tqdm

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # noqa

from metasyn.metaframe import MetaFrame
from metasyn.privacy import get_privacy
from metasyn.provider import DistributionProviderList
from metasyn.var import MetaVar


class VarConfig():
    def __init__(self, name: str, var_type: Optional[str] = None,
                 distribution: Optional[dict] = None,
                 prop_missing: Optional[float] = None,
                 description: Optional[str] = None,
                 data_free: Optional[bool] = None):
        self.name = name
        self.var_type = var_type
        self.distribution = distribution
        self.prop_missing = prop_missing
        self.description = description
        self.data_free = data_free

class MetaConfig():
    def __init__(self, var_configs: list,
                 dist_providers: list,
                 privacy: dict,
                 n_rows: Optional[int] = None):
        self.var_configs = var_configs
        self.dist_providers = dist_providers
        self.privacy = privacy
        self.n_rows = n_rows

    @staticmethod
    def get_var_type(series: pl.Series) -> str:
        """Convert polars dtype to metasyn variable type.

        This method uses internal polars methods, so this might break at some
        point.

        Parameters
        ----------
        series:
            Series to get the metasyn variable type for.

        Returns
        -------
        var_type:
            The variable type that is found.
        """
        try:
            polars_dtype = pl.datatypes.dtype_to_py_type(series.dtype).__name__
        except NotImplementedError:
            polars_dtype = pl.datatypes.dtype_to_ffiname(series.dtype)

        convert_dict = {
            "int": "discrete",
            "float": "continuous",
            "date": "date",
            "datetime": "datetime",
            "time": "time",
            "str": "string",
            "categorical": "categorical"
        }
        try:
            return convert_dict[polars_dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported polars type '{polars_dtype}") from exc

    def fit_var(self, series, provider_list, default_privacy):
        name = series.name
        var_config = self.var.get(name, {})
        dist_spec = var_config.get("distribution", None)

        if "privacy" in var_config:
            default_privacy = get_privacy(**var_config["privacy"])

        var_type = self.get_var_type(series)
        privacy = var_config.get("privacy", default_privacy)
        unique = var_config.get("unique", None)
        fit_kwargs = None if dist_spec is None else dist_spec.get("fit_kwargs", None)
        description = var_config.get("description", None)
        distribution = provider_list.fit(series, var_type, dist_spec, privacy, unique, fit_kwargs)
        prop_missing = (len(series) - len(series.drop_nulls())) / len(series)

        if var_config.get("data_free", False):
            raise ValueError(f"Variable '{name}' was found in the dataframe, but was assumed to be"
                             " data free in the configuration (file). Either remove the column from"
                             " the dataset or set the variable not to be data free.")

        return MetaVar(name, var_type, distribution, prop_missing, str(series.dtype), description)

    def data_free_var(self, col_name, provider_list):
        var_spec = self.var[col_name]
        if not var_spec.get("data_free", False):
            raise ValueError(f"Column with name '{var_spec}' not found and not declared as "
                             "data_free.")
        unique = var_spec.get("unique", False)
        distribution = provider_list.create(var_spec["var_type"], var_spec["distribution"], unique)
        prop_missing = var_spec.get("prop_missing", 0.0)
        description = var_spec.get("description", None)
        return MetaVar(col_name, var_spec["var_type"], distribution, prop_missing,
                       description=description)

    def generate_metaframe(self, df: Optional[pl.DataFrame] = None, progress_bar: bool = True):
        provider_list =  DistributionProviderList(self.dist_providers)
        default_privacy = get_privacy(**self.privacy)
        all_vars = []

        # First generate variables that exist in the dataframe
        if df is not None:
            self.n_rows = len(df) if self.n_rows is None else self.n_rows
            for col_name in tqdm(df.columns, disable=not progress_bar):
                new_var = self.fit_var(df[col_name], provider_list, default_privacy)
                all_vars.append(new_var)
            data_free_cols = set(self.var) - set(df.columns)

        else:
            if self.n_rows is None:
                raise ValueError("Error generating metaframe without dataframe: set the number of "
                                 "rows with the n_rows variable under the 'general' section.")
            data_free_cols = set(self.var)

        for col_name in data_free_cols:
            new_var = self.data_free_var(col_name, provider_list)
            all_vars.append(new_var)
        return MetaFrame(all_vars, self.n_rows)

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
