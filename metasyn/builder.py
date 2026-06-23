import inspect
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from metasyn.distribution.base import BaseDistribution, BaseFitter, DistributionLike
from metasyn.metaframe import MetaFrame
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.util import get_var_type
from metasyn.var import MetaVar

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa


class VarBuilder():
    def __init__(self, series: pl.Series | None = None,
                 name: str | None = None,
                 mf_builder: Optional["MetaFrameBuilder"] = None,
                 prop_missing: float | None = None,
                 privacy: BasePrivacy | None = BasicPrivacy(),
                 distribution: str | DistributionLike | dict | None = None,
                 fitter: BaseFitter | None = None,
                 description: str | None = None):
        if series is not None:
            series = pl.Series(series)
        self.series = series
        self.name = series.name if name is None and series is not None else name
        self.prop_missing = prop_missing
        self._privacy = privacy
        self.mf_builder = mf_builder
        self._distribution: str | DistributionLike | dict | None = distribution
        self.fitter = fitter
        self.description: str | None = description
        self.plugins = None
        self._var_type = None

    @property
    def registry(self):
        plugins = self.plugins
        if plugins is None:
            plugins = getattr(getattr(self, "mf_builder", None), "plugins", None)

        return DistributionRegistry.parse(plugins)

    @property
    def privacy(self):
        if self.mf_builder is not None and self._privacy is None:
            return self.mf_builder.defaults.get("privacy", None)
        return self._privacy

    @privacy.setter
    def privacy(self, value):
        self._privacy = value

    @property
    def distribution(self):
        if self.mf_builder is not None and self._distribution is None and self._var_type is not None:
            return self.mf_builder.get_default_distribution(self._var_type)
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        self._distribution = value

    @privacy.setter
    def privacy(self, value):
        self._distribution = value

    @property
    def var_type(self):
        if self._var_type is not None:
            return self._var_type
        distribution = {} if self.distribution is None else self.distribution

        series_var_type = None if self.series is None else get_var_type(self.series)

        if isinstance(distribution, dict):
            return distribution.get("var_type", series_var_type)
        return series_var_type

    @var_type.setter
    def var_type(self, value: str):
        self._var_type = value

    @property
    def dtype(self):
        if self.series is not None:
            return str(self.series.dtype)
        return "unknown"

    @property
    def recipe(self):
        avail_recipes = [DistributionRecipe, FitterRecipe, FindDistributionRecipe,
                         UnqFindDistributionRecipe]
        for recipe_class in avail_recipes:
            recipe = recipe_class.create(self)
            if recipe is not None:
                return recipe
        raise TypeError(f"Unknown type for recipe: {type(self.distribution)}.")

    def get_creation_method(self, fitter: BaseFitter | str | None) -> dict:
        """Create a dictionary on how the distribution was created.

        Parameters
        ----------
        privacy
            Privacy object with which the dictionary is being created.

        Returns
        -------
            Dictionary containing all the non-default settings for the creation method.
        """
        if isinstance(fitter, str):
            ret_dict: dict[str, Any] = {"created_by": fitter}
        else:
            ret_dict: dict[str, Any] = {"created_by": "metasyn"}
        if isinstance(self.distribution, dict):
            dist_dict = {var: self.distribution.get(var) for var in ["name", "unique", "parameters", "version"]
                         if var in self.distribution}
            if len(dist_dict) != 0:
                ret_dict["distribution"] = dist_dict
        fit_dict = {}
        if fitter is not None and isinstance(fitter, BaseFitter):
            fit_dict = fitter.to_dict()

        if len(fit_dict) != 0:
            ret_dict["fitter"] = fit_dict

        return ret_dict

    def fit(self):
        if self.prop_missing is None:
            prop_missing = (len(self.series) - len(self.series.drop_nulls())) / len(self.series)
        else:
            prop_missing = self.prop_missing
        dist, fitter = self.recipe.fit()
        return MetaVar(self.name, self.var_type, dist, self.dtype, self.description,
                       prop_missing, creation_method=self.get_creation_method(fitter))

    def _find_fitters(self, unique):
        distribution = {} if self.distribution is None else deepcopy(self.distribution)
        distribution.update({"privacy": self.privacy})
        var_type = distribution.pop("var_type", get_var_type(self.series))
        distribution.pop("unique", None)
        distribution["var_type"] = var_type
        fitter_class = self.registry.filter_fitters(**distribution, unique=unique)
        fitters = [f(self.privacy) for f in fitter_class]
        return fitters


class MetaFrameBuilder():
    def __init__(self, name="single"):
        self.file_format = None
        self.columns = []
        self.var_builders = {}
        self.n_rows = None
        self.plugins = None
        self.name = name
        self.defaults = {}

    def __getitem__(self, item: str):
        return self.var_builders[item]

    def add_dataframe(self, df, file_format):
        # self.df = df
        self.columns = df.columns

        self.file_format = file_format
        for col in df.columns:
            self.var_builders[col] = VarBuilder(df[col], col, self)
        self.n_rows = len(df) if self.n_rows is None else self.n_rows

    def add_column(self, name):
        self.var_builders[name] = VarBuilder(None, name, self)
        self.columns.append(name)
        self.var_builders[name].prop_missing = 0.0

    def add_config(self, config: Path | str | dict) -> "MetaFrameBuilder":
        if isinstance(config, (Path, str)):
            try:
                with open(config, "rb") as handle:
                    config = tomllib.load(handle)
            except FileNotFoundError as fnf_error:
                raise FileNotFoundError(f"It appears '{config}' is not a valid filepath."
                                        f" Please provide a path to a .toml file to load a MetaConfig"
                                        f" from.") from fnf_error
            except tomllib.TOMLDecodeError as value_error:
                if Path(config).suffix != ".toml":
                    raise ValueError(f"It appears '{Path(config).name}' is a"
                                    f" '{Path(config).suffix}' file."
                                    f" To load a MetaConfig, "
                                    f"provide the configuration as a .toml file.") from value_error
                raise value_error

        config_version = config.get("config_version", "2.0")

        for parser in [ConfigV1XParser()]:
            if config_version in parser.supports:
                parser.read_dict(config, self)
                return
        raise ValueError(f"Cannot read configuration file, because version {config_version} is not "
                         "supported.")

    def get_default_distribution(self, var_type):
        # print(self.defaults.get("distribution", {}).get(var_type, None))
        return self.defaults.get("distribution", {}).get(var_type, None)

    def set_privacy(self, privacy):
        if isinstance(privacy, dict):
            for name, priv in privacy.items():
                self[name].privacy = priv
        else:
            self.privacy = privacy

    def preview(self, n_row_synthesize: int = 10, n_row_fit: None | int = None):
        pass

    def fit(self, progress_bar: bool = True):
        vars = []
        for col in tqdm(self.columns, disable=not progress_bar):
            vars.append(self.var_builders[col].fit())
        return MetaFrame(vars, self.n_rows, self.file_format, self.name)

class ConfigV1XParser():
    keys = ["n_rows", "config_version", "file", "privacy", "defaults", "plugins",
            "var"]
    supports = ["1.0", "1.1", "1.2"]

    def read_dict(self, config_dict: dict, builder: MetaFrameBuilder):
        config_dict = deepcopy(config_dict)
        if not set(config_dict.keys()) <= set(self.keys):
            raise ValueError(f"Error parsing configuration."
                             f" Unknown keys detected: '{list(config_dict)}'")

        for var_dict in config_dict.get("var", []):
            if var_dict.get("data_free", config_dict.get("defaults", {}).get("data_free", False)):
                builder.add_column(var_dict["name"])
            for attr, val in var_dict.items():
                setattr(builder[var_dict["name"]], attr, val)

        if "n_rows" in config_dict:
            builder.n_rows = config_dict["n_rows"]
        if "plugins" in config_dict:
            builder.plugins = config_dict["plugins"]
        if "file" in config_dict:
            builder.file_format = config_dict["file"]
        if "privacy" in config_dict and "defaults" in config_dict:
            raise ValueError("Error parsing configuration file: cannot have both [privacy]"
                                 " and [defaults] tables.")
        if "privacy" in config_dict:
            builder.defaults["privacy"] = config_dict["privacy"]
        if "defaults" in config_dict:
            builder.defaults = config_dict["defaults"]


@dataclass
class DistributionRecipe():
    distribution: BaseDistribution

    def fit(self):
        return self.distribution, "user"

    @classmethod
    def create(cls, var_builder: VarBuilder):
        if isinstance(var_builder.distribution, BaseDistribution):
            if var_builder.series is None:
                var_builder.series = pl.Series([var_builder.distribution.draw()])
            return cls(var_builder.distribution)
        elif isinstance(var_builder.distribution, dict) and "parameters" in var_builder.distribution:
            dist_class = var_builder.registry.find_distribution(
                var_builder.distribution.get("name"),
                var_builder.var_type,
                var_builder.distribution.get("unique", False),
                var_builder.distribution.get("version", None))
            dist = dist_class(**var_builder.distribution["parameters"])
            if var_builder.series is None:
                var_builder.series = pl.Series([dist.draw()])
            return cls(dist)
        return None

@dataclass
class FitterRecipe():
    series: pl.Series
    fitter: BaseFitter

    def fit(self):
        return self.fitter.fit(self.series), self.fitter

    @classmethod
    def create(cls, var_builder: VarBuilder):
        if isinstance(var_builder.fitter, BaseFitter):
            return cls(var_builder.series, var_builder.fitter)
        return None


@dataclass
class FindDistributionRecipe():
    series: pl.Series
    fitters: list[BaseFitter]

    def fit(self):
        if len(self.fitters) == 1:
            return FitterRecipe(self.series, self.fitters[0]).fit()
        if len(self.fitters) == 0:
            raise ValueError()

        return self.fit_with_bic(self.series)[:2]

    def fit_with_bic(self, series):
        distributions = [f.fit(series) for f in self.fitters]
        bic = [d.information_criterion(series) for d in distributions]
        return distributions[np.argmin(bic)], self.fitters[np.argmin(bic)], np.min(bic)

    @classmethod
    def create(cls, var_builder: VarBuilder):
        dist = var_builder.distribution
        if var_builder.series is None:
            return None
        if isinstance(dist, dict) and dist.get("unique", None) is not None:
            fitters = var_builder._find_fitters(dist["unique"])
            return cls(var_builder.series, fitters)
        if inspect.isclass(dist) and issubclass(dist, DistributionLike):
            fitter_classes = var_builder.registry.find_fitters(dist.name, var_type=var_builder.var_type)
            return cls(var_builder.series, [f(var_builder.privacy) for f in fitter_classes])
        return None

@dataclass
class UnqFindDistributionRecipe():
    series: pl.Series
    fitters: list[BaseFitter]
    unq_fitters: list[BaseFitter]

    def fit(self):
        series = self.series
        dist, fitter, bic = FindDistributionRecipe(self.series, self.fitters).fit_with_bic(series)
        if len(self.unq_fitters) == 0:
            return dist, fitter
        unq_dist, unq_fitter, unq_bic = FindDistributionRecipe(self.series, self.unq_fitters).fit_with_bic(series)
        if unq_bic + 16 < bic:
            if unq_dist.name == "core.unique_key" and unq_dist.consecutive:
                return unq_dist, unq_fitter
            warnings.warn(
                f"\nMetasyn detected that variable '{series.name}' is potentially unique.\n"
                f"Use var_spec=[VarSpec(\"{series.name}\", unique=True)] to make it unique."
                f"\nTo dismiss this warning use [VarSpec(\"{series.name}\", unique=False)]."
                "\nIf you are using a configuration file add distribution = {unique = True}"
                f" for the variable with name '{series.name}'.",
                UserWarning
            )
        return dist, fitter

    @classmethod
    def create(cls, var_builder: VarBuilder):
        dist = var_builder.distribution
        if dist is None or (isinstance(dist, dict) and dist.get("unique", None) is None) and var_builder.series is not None:
            fitters = var_builder._find_fitters(False)
            unq_fitters = var_builder._find_fitters(True)
            return cls(var_builder.series, fitters, unq_fitters)

