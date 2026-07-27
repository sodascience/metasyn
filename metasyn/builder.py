"""Builder classes to build your metaframe one step at a time."""
from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractclassmethod, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from metasyn.distribution.base import BaseFitter, DistributionLike, VarLog
from metasyn.file import BaseFileInterface
from metasyn.metaframe import MetaFrame
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.util import get_var_type
from metasyn.var import MetaVar

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa


class FitLog():
    """Logbook for the builder fitting process."""

    def __init__(self):
        self.log = {}

    def write_csv(self, fp: Path | str):
        """Create a CSV file containing fitting data for each column.

        Parameters
        ----------
        fp:
            File to write to.
        """
        self.to_dataframe().write_csv(fp)

    def write_md(self, fp: Path | str):
        md_str = "# Fit log\n"
        for key in self.log:
            md_str += self.log[key].to_md(key) + "\n"
        with open(fp, "w", encoding="utf-8") as handle:
            handle.write(md_str)

    def __getitem__(self, key: str) -> VarLog:
        return self.log[key]

    def __setitem__(self, key: str, val: VarLog):
        self.log[key] = val

    def add_col(self, name: str):
        """Create fit log entry for column.

        Parameters
        ----------
        name:
            Name of the column to add.
        """
        self.log[name] = VarLog()

    def reset(self):
        """Remove all logs, but keep columns."""
        for key in self.log:
            self.log[key] = VarLog()

    # def print(self):
        # pass
        # print("\n\n".join(str(x) for x in self.log.values()))

    def to_dataframe(self) -> pl.DataFrame:
        """Create dataframe from fit log.

        Returns
        -------
        df:
            Dataframe containing all entries in the log.
        """
        data_dict = defaultdict(list)
        data_dict["column"] = list(self.log)

        for var_name, var_log in self.log.items():
            for key, val in var_log.to_dict().items():
                data_dict[key].append(val)
        return pl.DataFrame(data_dict)


class VarBuilder():
    """MetaVar builder class to set the details of the distribution.

    MetaVar instances will be generated step by step.

    Parameters
    ----------
    series:
        Series/column that contains the data to fit the distributions to. Can
        be None if the distribution is completely determined otherwise.
    name:
        Name of the column.
    mf_builder:
        MetaFrame builder. This is mainly used to get default values for the whole
        synthetic dataframe.
    prop_missing:
        Proportion of missing values. If None, get this proportion from the original data.
    privacy:
        Privacy object which will be used for the fitting method.
    distribution:
        Distribution to be used. If None, find the distribution with the best statistical
        fit.
    fitter:
        Currently unused.
    description:
        Description attached to the column.
    """

    def __init__(self, series: pl.Series | None | DistributionLike = None,
                 name: str | None = None,
                 mf_builder: Optional["MetaFrameBuilder"] = None,
                 prop_missing: float | None = None,
                 privacy: BasePrivacy | None = None,
                 distribution: str | DistributionLike | dict | None = None,
                 fitter: BaseFitter | None = None,
                 description: str | None = None,
                 var_type: str | None = None,
                 hidden: bool = False):
        # if series is not None:
            # series = pl.Series(series)
        self._series: pl.Series | DistributionLike | None = None
        self.series = series
        self.name = series.name if name is None and isinstance(series, pl.Series) else name
        self.prop_missing = prop_missing
        self._privacy = privacy
        self.mf_builder = mf_builder
        self._distribution: str | DistributionLike | dict | None = distribution
        self.fitter = fitter
        self.description: str | None = description
        self.plugins: None | DistributionRegistry | str | list[str] = None
        self._var_type: str | None = var_type
        self.hidden = hidden

    @property
    def registry(self) -> DistributionRegistry:
        """Distribution registry to be used for finding distributions and fitters."""
        plugins = self.plugins
        if plugins is None:
            plugins = getattr(getattr(self, "mf_builder", None), "plugins", None)

        return DistributionRegistry.parse(plugins)

    @registry.setter
    def registry(self, value: None | DistributionRegistry | str | list[str]):
        self.plugins = value

    @property
    def series(self) -> pl.Series | None:
        if self._series is None:
            return None
        if isinstance(self._series, DistributionLike):
            if self.mf_builder is None:
                raise ValueError("Cannot create series without correspondig Metaframe builder.")
            synth_dict = {name: self.mf_builder[name].series for name in self._series.dependencies}
            return pl.Series(self._series.draw_list(self.mf_builder.n_rows, synth_dict))
        return pl.Series(self._series)

    @series.setter
    def series(self, value: pl.Series | DistributionLike | None):
        self._series = value

    @property
    def privacy(self) -> BasePrivacy:
        if self.mf_builder is not None and self._privacy is None:
            return self.mf_builder.defaults.get("privacy", BasicPrivacy())
        return self._privacy if self._privacy is not None else BasicPrivacy()

    @privacy.setter
    def privacy(self, value: BasePrivacy | None):
        self._privacy = value

    @property
    def distribution(self) -> None | dict | str | DistributionLike:
        """Distribution directives for fitting a distribution."""
        if (self.mf_builder is not None and self._distribution is None
                and self.var_type is not None):
            return self.mf_builder.get_default_distribution(self.var_type)
        return self._distribution

    @distribution.setter
    def distribution(self, value: None | dict | str | DistributionLike):
        self._distribution = value

    @property
    def var_type(self) -> str | None:
        """Variable type for the distribution."""
        if self._var_type is not None:
            return self._var_type
        distribution = {} if self._distribution is None else self._distribution

        series_var_type = None if self.series is None else get_var_type(self.series)

        if isinstance(distribution, dict):
            var_type = distribution.get("var_type", series_var_type)
            assert var_type is None or isinstance(var_type, str)
            return distribution.get("var_type", series_var_type)
        return series_var_type

    @var_type.setter
    def var_type(self, value: str):
        self._var_type = value

    @property
    def dtype(self) -> str:
        """Type of the elements in the series to be generated."""
        if self.series is not None:
            return str(self.series.dtype)
        return "unknown"

    @property
    def recipe(self) -> BaseRecipe:
        """Get recipe to create a distribution."""
        avail_recipes: list[type[BaseRecipe]] = [
            DistributionRecipe, FitterRecipe, FindDistributionRecipe, UnqFindDistributionRecipe]
        for recipe_class in avail_recipes:
            recipe = recipe_class.create(self)  # type: ignore
            if recipe is not None:
                return recipe
        raise TypeError(f"Cannot find recipe for distribution {self.distribution}, wrong type?.")

    def fit(self, fit_log: VarLog | None = None):
        """Fit or create the distribution.

        Returns
        -------
            A MetaVar with the fitted distribution.
        """
        if self.prop_missing is None and self.series is not None:
            prop_missing = (len(self.series) - len(self.series.drop_nulls())) / len(self.series)
        elif self.prop_missing is None:
            prop_missing = 0.0
        else:
            prop_missing = self.prop_missing
        if self.var_type is None:
            raise ValueError("Could not detect variable type, please set variable type for "
                             f"'{self.name}'.")
        if fit_log is None:
            fit_log = VarLog()
        dist = self.recipe.fit(fit_log)
        return MetaVar(self.name, self.var_type, dist, self.dtype, self.description,
                       prop_missing, creation_method=fit_log.creation_method,
                       hidden=self.hidden)

    def _find_fitters(self, unique):
        distribution = {} if self.distribution is None else deepcopy(self.distribution)
        distribution = {"name": distribution} if isinstance(distribution, str) else distribution
        distribution.update({"privacy": self.privacy})
        var_type = distribution.pop("var_type", get_var_type(self.series))
        distribution.pop("unique", None)
        distribution["var_type"] = var_type
        fitter_class = self.registry.filter_fitters(**distribution, unique=unique)
        fitters = [f(self.privacy) for f in fitter_class]
        return fitters


class MetaFrameBuilder():
    """Builder class for creating metaframes.

    This class allows you to build your metaframe step by step instead of in one go with the
    ``MetaFrame.fit_dataframe()` method.

    Parameters
    ----------
    name:
        Name of the dataframe. Only really used when you are synthesizing multiple datasets at
        the same time.
    """

    def __init__(self, name="single"):
        self.file_format = None
        self.columns = []
        self.var_builders = {}
        self.n_rows = None
        self.plugins = None
        self.name = name
        self.defaults = {}
        self.fit_log = FitLog()

    def __getitem__(self, item: str):
        return self.var_builders[item]

    def add_dataframe(self, df: pl.DataFrame, file_format: BaseFileInterface | dict | None = None):
        """Add a dataframe to the metaframe builder.

        Parameters
        ----------
        df
            DataFrame to be synthesized.
        file_format
            File format description linked to the dataframe.
        """
        self.columns = df.columns

        self.file_format = file_format
        for col in df.columns:
            self.var_builders[col] = VarBuilder(df[col], col, self)
            self.fit_log.add_col(col)
        self.n_rows = len(df) if self.n_rows is None else self.n_rows

    def add_column(self, name: str, hidden: bool = False, var_type: str | None = None):
        """Add a new column to the MetaFrame being built.

        Parameters
        ----------
        name
            Name of the new column.
        """
        self.var_builders[name] = VarBuilder(None, name, self, hidden=hidden, var_type=var_type)
        self.fit_log.add_col(name)
        self.columns.append(name)

    def add_config(self, config: Path | str | dict) -> "MetaFrameBuilder":
        """Configure the MetaFrame from a configuration file.

        This configuration file can have new columns that are declared data free.

        Parameters
        ----------
        config:
            Configuration file or dictionary that will be applied to the MetaFrame.
        """
        if isinstance(config, (Path, str)):
            try:
                with open(config, "rb") as handle:
                    config_dict: dict = tomllib.load(handle)
            except FileNotFoundError as fnf_error:
                raise FileNotFoundError(
                    f"It appears '{config}' is not a valid filepath."
                    f" Please provide a path to a .toml file to load a MetaConfig"
                    f" from.") from fnf_error
            except tomllib.TOMLDecodeError as value_error:
                if Path(config).suffix != ".toml":
                    raise ValueError(f"It appears '{Path(config).name}' is a"
                                    # f" '{Path(config).suffix}' file."
                                    f" To load a MetaConfig, "
                                    f"provide the configuration as a .toml file.") from value_error
                raise value_error
        else:
            config_dict = config
        config_version = config_dict.get("config_version", "2.0")

        for parser in [ConfigV1XParser()]:
            if config_version in parser.supports:
                parser.read_dict(config_dict, self)
                return self
        raise ValueError(f"Cannot read configuration file, because version {config_version} is not "
                         "supported.")

    def get_default_distribution(self, var_type) -> str | dict | None | DistributionLike:
        """Get the default distribution for a variable type.

        Parameters
        ----------
        var_type
            Variable type such as "string", "discrete", "categorical".

        Returns
        -------
            The default distribution.
        """
        return self.defaults.get("distribution", {}).get(var_type, None)

    @property
    def privacy(self) -> BasePrivacy:
        """The default privacy for all columns."""
        return self.defaults.get("privacy", BasicPrivacy())

    @privacy.setter
    def privacy(self, value: BasePrivacy):
        self.defaults.update({"privacy": value})

    # def preview(self, n_row_synthesize: int = 10, n_row_fit: None | int = None):
        # pass

    def fit(self, progress_bar: bool = True) -> MetaFrame:
        """Create a MetaFrame from the builder.

        Parameters
        ----------
        progress_bar:
            Whether to display a progress bar.
        """
        vars = []
        self.fit_log.reset()
        for col in tqdm(self.columns, disable=not progress_bar):
            vars.append(self.var_builders[col].fit(self.fit_log[col]))
        return MetaFrame(vars, self.n_rows, self.file_format, self.name)

class ConfigV1XParser():
    """TOML confifuration parser for versions 1.0, 1.1 and 1.2."""

    keys = ["n_rows", "config_version", "file", "privacy", "defaults", "plugins",
            "var"]
    supports = ["1.0", "1.1", "1.2"]

    def read_dict(self, config_dict: dict, builder: MetaFrameBuilder):
        """Read a dictionary containing the configuration.

        Parameters
        ----------
        config_dict
            Configuration dictionary to parse.
        builder
            MetaFrame Builder to adjust from the configuration dictionary.

        Raises
        ------
        ValueError
            If unknown keys are detected or if there are both privacy and defaults sections.
        """
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


class BaseRecipe(ABC):
    """Base class for distribution recipes."""

    @abstractmethod
    def fit(self, fit_log: VarLog) -> DistributionLike:
        """Use the recipe to fit or get the correct distribution."""
        pass

    @abstractclassmethod  # type: ignore
    def create(cls, var_builder: VarBuilder) -> BaseRecipe | None:
        """Create a recipe from the information in the var builder.

        If the recipe cannot build itself from the provided information, None will be returned.
        """

@dataclass
class DistributionRecipe(BaseRecipe):
    """Distribution recipe without any fitting.

    Used for example if you use builder["col"].distribution = DiscreteUniformDistribution(0, 1).
    """

    distribution: DistributionLike

    def fit(self, fit_log):
        fit_log.add(created_by="user")
        fit_log.add(recipe="User defined distribution.")
        return self.distribution

    @classmethod
    def create(cls, var_builder: VarBuilder):
        if isinstance(var_builder.distribution, DistributionLike):
            # if var_builder.series is None:
                # var_builder.series = pl.Series([var_builder.distribution.draw()])
            return cls(var_builder.distribution)
        elif (isinstance(var_builder.distribution, dict)
                and "parameters" in var_builder.distribution):
            name = var_builder.distribution.get("name")
            if name is None or not isinstance(name, str):
                raise TypeError("Distribution dictionary with parameters, but name missing or "
                                f"wrong type for column {var_builder.name}.")
            dist_class = var_builder.registry.find_distribution(
                var_builder.distribution["name"],
                var_builder.var_type,
                var_builder.distribution.get("unique", False),
                var_builder.distribution.get("version", None))
            dist = dist_class(**var_builder.distribution["parameters"])
            # if var_builder.series is None:
                # var_builder.series = pl.Series([dist.draw()])
            return cls(dist)
        return None

@dataclass
class FitterRecipe(BaseRecipe):
    """Recipe where one candidate fitter is used.

    For example when builder["col"].distribution = "uniform".
    """

    series: pl.Series
    fitter: BaseFitter

    def fit(self, fit_log: VarLog):
        fit_log.add(recipe="Single fitter.")
        fit_log.add(created_by="metasyn")
        fit_log.add(fitter=self.fitter)
        return self.fitter.fit(self.series, fit_log)

    @classmethod
    def create(cls, var_builder: VarBuilder):
        if isinstance(var_builder.fitter, BaseFitter) and var_builder.series is not None:
            return cls(var_builder.series, var_builder.fitter)
        return None


@dataclass
class FindDistributionRecipe(BaseRecipe):
    """Recipe where multiple fitters are considered."""

    series: pl.Series
    fitters: list[BaseFitter]

    def fit(self, fit_log):
        fit_log.add(created_by="metasyn")
        fit_log.add(recipe="Fit multiple candidate distributions: "
                    f"({', '.join(f.__class__.__name__ for f in self.fitters)}).")
        dist, fit_dict = self.fit_with_bic(self.series)
        fit_log.extend(fit_dict)
        return dist

    def fit_with_bic(self, series):
        fit_logs = [VarLog() for _ in range(len(self.fitters))]
        distributions = [f.fit(series, f_log) for f, f_log in zip(self.fitters, fit_logs)]
        bic = [d.information_criterion(series) for d in distributions]
        return (distributions[np.argmin(bic)],
                VarLog().extend(fit_logs[np.argmin(bic)]).add(bic=min(bic)).add(fitter=self.fitters[np.argmin(bic)]))

    @classmethod
    def create(cls, var_builder: VarBuilder):
        dist = var_builder.distribution
        if var_builder.series is None:
            return None
        if isinstance(dist, dict) and dist.get("unique", None) is not None:
            fitters = var_builder._find_fitters(dist["unique"])
            return cls(var_builder.series, fitters)
        if inspect.isclass(dist) and issubclass(dist, DistributionLike):
            fitter_classes = var_builder.registry.find_fitters(dist.name,
                                                               var_type=var_builder.var_type)
            fitters = [f(var_builder.privacy) for f in fitter_classes]
            if len(fitters) == 1:
                return FitterRecipe(var_builder.series, fitters[0])
            elif len(fitters) > 1:
                return cls(var_builder.series, fitters)
        return None

@dataclass
class UnqFindDistributionRecipe(BaseRecipe):
    """Recipe where we consider both unique and non-unique fitters."""

    series: pl.Series
    fitters: list[BaseFitter]
    unq_fitters: list[BaseFitter]

    def fit(self, fit_log):
        fit_log.add(created_by="metasyn")
        fit_log.add(recipe="Fit multiple candidate distributions, unique and non-unique "
                    "(for warnings): "
                     + ", ".join([x.__class__.__name__ for x in self.fitters + self.unq_fitters]))
        series = self.series
        dist, non_fit_log = FindDistributionRecipe(self.series, self.fitters).fit_with_bic(series)
        unq_dist, unq_fit_log = FindDistributionRecipe(self.series,
                                                               self.unq_fitters).fit_with_bic(series)
        if unq_fit_log.bic[-1] + 16 < non_fit_log.bic[-1]:
            if unq_dist.name == "core.unique_key" and unq_dist.consecutive:
                fit_log.extend(unq_fit_log)
                return unq_dist
            warnings.warn(
                f"\nMetasyn detected that variable '{series.name}' is potentially unique.\n"
                f"Use var_spec=[VarSpec(\"{series.name}\", unique=True)] to make it unique."
                f"\nTo dismiss this warning use [VarSpec(\"{series.name}\", unique=False)]."
                "\nIf you are using a configuration file add distribution = {unique = True}"
                f" for the variable with name '{series.name}'.",
                UserWarning
            )
        fit_log.extend(non_fit_log)
        return dist

    @classmethod
    def create(cls, var_builder: VarBuilder):
        dist = var_builder.distribution
        if ((
                isinstance(dist, str)
                or dist is None
                or (isinstance(dist, dict) and dist.get("unique", None) is None))
                    and var_builder.series is not None):
            fitters = var_builder._find_fitters(False)
            unq_fitters = var_builder._find_fitters(True)
            if len(unq_fitters) == 0:
                if len(fitters) == 1:
                    return FitterRecipe(var_builder.series, fitters[0])
                return FindDistributionRecipe(var_builder.series, fitters)
            if len(fitters) == 0:
                if len(unq_fitters) == 1:
                    return FitterRecipe(var_builder.series, unq_fitters[0])
                return FindDistributionRecipe(var_builder.series, unq_fitters)
            return cls(var_builder.series, fitters, unq_fitters)
