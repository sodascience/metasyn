import inspect
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from tqdm import tqdm

from metasyn.distribution.base import BaseDistribution, BaseFitter, DistributionLike
from metasyn.metaframe import MetaFrame
from metasyn.registry import DistributionRegistry
from metasyn.util import get_var_type
from metasyn.var import MetaVar


class VarBuilder():
    def __init__(self, name, mf_builder):
        self.name = name
        self.series = None
        self.prop_missing = None
        self._privacy = None

        # self.unique = None
        self.mf_builder = mf_builder
        self.distribution: str | DistributionLike | dict | None = None
        self.description: str | None = None

    @property
    def registry(self):
        return DistributionRegistry.parse(getattr(self, "plugins", self.mf_builder.plugins))

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
        return str(self.series.dtype)

    @property
    def recipe(self):
        if isinstance(self.distribution, DistributionLike):
            return DistributionRecipe(self.distribution)
        if isinstance(self.distribution, dict) or self.distribution is None:
            if self.distribution is None or self.distribution.get("unique", None) is None:
                fitters = self._find_fitters(False)
                unq_fitters = self._find_fitters(True)
                return UnqFindDistributionRecipe(self.series, fitters, unq_fitters)
            fitters = self._find_fitters(self.distribution["unique"])
            return FindDistributionRecipe(self.series, fitters)
        if inspect.isclass(self.distribution) and issubclass(self.distribution, DistributionLike):
            return FindDistributionRecipe(self.series, self.registry.find_fitter(self.distribution))
        raise TypeError(f"Unknown type for recipe: {type(self.distribution)}.")

    def fit(self):
        if self.prop_missing is None:
            prop_missing = (len(self.series) - len(self.series.drop_nulls())) / len(self.series)
        return MetaVar(self.name, self.var_type, self.recipe.fit()[0], self.dtype, self.description,
                       prop_missing, creation_method="metasyn")

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

    def set_privacy(self, privacy):
        if isinstance(privacy, dict):
            for name, priv in privacy.items():
                self[name].privacy = priv
        else:
            self.privacy = privacy

    def preview(self):
        pass

    def fit(self):
        vars = []
        for col in tqdm(self.columns):
            vars.append(self.var_builders[col].fit())
        return MetaFrame(vars, self.n_rows, self.file_format, self.name)


@dataclass
class DistributionRecipe():
    distribution: BaseDistribution

    def fit(self):
        return self.distribution, "Set by user"


@dataclass
class FitterRecipe():
    series: pl.Series
    fitter: BaseFitter
    fit_kwargs: dict = field(default_factory=dict)

    def fit(self):
        return self.fitter.fit(self.series, **self.fit_kwargs), self.fitter

@dataclass
class FindDistributionRecipe():
    series: pl.Series
    fitters: list[BaseFitter]

    def fit(self):
        if len(self.fitters) == 1:
            return FitterRecipe(self.series, self.fitters[0], {}).fit()
        if len(self.fitters) == 0:
            raise ValueError()

        return self.fit_with_bic(self.series)[:2]

    def fit_with_bic(self, series):
        distributions = [f.fit(series) for f in self.fitters]
        bic = [d.information_criterion(series) for d in distributions]
        return distributions[np.argmin(bic)], self.fitters[np.argmin(bic)], np.min(bic)

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



# class ColumnReferenceRecipe():
#     reference: ColumnReference
    

