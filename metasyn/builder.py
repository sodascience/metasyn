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
        self.distribution: str | DistributionLike | dict | None = distribution
        self.fitter = fitter
        self.description: str | None = description
        self.plugins = None

    @property
    def registry(self):
        plugins = self.plugins
        if plugins is None:
            plugins = getattr(getattr(self, "mf_builder", None), "plugins", None)

        return DistributionRegistry.parse(plugins)

    @property
    def privacy(self):
        if self._privacy is None:
            return self.mf_builder.privacy
        return self._privacy

    @privacy.setter
    def privacy(self, value):
        self._privacy = value

    @property
    def var_type(self):
        distribution = {} if self.distribution is None else self.distribution
        if isinstance(distribution, dict):
            return distribution.get("var_type", get_var_type(self.series))
        return get_var_type(self.series)

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
        # if isinstance(self.distribution, DistributionLike):
        #     return DistributionRecipe(self.distribution)
        # if isinstance(self.distribution, dict) or self.distribution is None:
        #     if self.distribution is not None and "parameters" in self.distribution:
        #         parameters = self.distribution.pop("parameters")
        #         dist_class = self.registry.find_distribution(**self.distribution,
        #                                                      var_type=self.var_type)
        #         return DistributionRecipe(dist_class(**parameters))
        #     if self.distribution is None or self.distribution.get("unique", None) is None:
        #         fitters = self._find_fitters(False)
        #         unq_fitters = self._find_fitters(True)
        #         return UnqFindDistributionRecipe(self.series, fitters, unq_fitters)
        #     fitters = self._find_fitters(self.distribution["unique"])
        #     return FindDistributionRecipe(self.series, fitters)
        # if inspect.isclass(self.distribution) and issubclass(self.distribution, DistributionLike):
            # return FindDistributionRecipe(self.series, self.registry.find_fitter(self.distribution))
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
        # self.df = None
        self.file_format = None
        self.default_privacy = None
        # self.override_privacy = {}
        self.privacy = None
        self.columns = []
        self.var_builders = {}
        self.n_rows = None
        self.plugins = None
        self.name = name

    def __getitem__(self, item: str):
        return self.var_builders[item]

    def add_dataframe(self, df, file_format):
        # self.df = df
        self.columns = df.columns

        self.file_format = file_format
        for col in df.columns:
            self.var_builders[col] = VarBuilder(col, self)
            self.var_builders[col].series = df[col]
        self.n_rows = len(df) if self.n_rows is None else self.n_rows

    def add_column(self, name):
        self.var_builders[name] = VarBuilder(name, self)
        self.columns.append(name)
        self.var_builders[name].prop_missing = 0.0

    def add_config(self, config: Path | str | dict) -> "MetaFrameBuilder":
        if isinstance(config, (Path, str)):
            with open(config, "rb") as handle:
                config = tomllib.load(handle)

        config_version = config.get("config_version", None)
        # try:
            # config_version = config["version"]
        # except KeyError:
            # raise ValueError("Configuration or configuration file does not contain a version number.")

        if config_version is None:
            raise warnings.warn("Unknown version of configuration file.", UserWarning)

        config = deepcopy(config["table"][0])

        # TODO: do some error checking on versions
        self.name = config.get("name", self.name)
        self.file_format = config.get("file_format", )

        if config["table_type"] == "dataframe":
            for var_config in config["var"]:
                col_name = var_config.pop("name")
                for attr, val in var_config.items():
                    setattr(self[col_name], attr, val)

    def set_privacy(self, privacy):
        if isinstance(privacy, dict):
            for name, priv in privacy.items():
                self[name].privacy = priv
        else:
            self.privacy = privacy

    def preview(self):
        pass

    def fit(self, progress_bar: bool = True):
        vars = []
        for col in tqdm(self.columns, disable=not progress_bar):
            vars.append(self.var_builders[col].fit())
        return MetaFrame(vars, self.n_rows, self.file_format, self.name)


@dataclass
class DistributionRecipe():
    distribution: BaseDistribution

    def fit(self):
        return self.distribution, "user"

    @classmethod
    def create(cls, var_builder: VarBuilder):
        if isinstance(var_builder.distribution, BaseDistribution):
            return cls(var_builder.distribution)
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

