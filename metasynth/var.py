"""Variable module that creates metadata variables."""  # pylint: disable=invalid-name

from __future__ import annotations

from typing import Union, Dict, Any, Optional

import polars as pl
import pandas as pd
import numpy as np

from metasynth.distribution.base import BaseDistribution
from metasynth.disttree import BaseDistributionTree, get_disttree


def _to_polars(series: Union[pd.Series, pl.Series]) -> pl.Series:
    if isinstance(series, pl.Series):
        return series
    if len(series.dropna()) == 0:
        series = pl.Series([None for _ in range(len(series))])
    else:
        series = pl.Series(series)
    return series


class MetaVar():
    """Meta data variable.

    Acts as a base class for specific types of variables, but also as a
    launching pad for detecting its type.

    Parameters
    ----------
    var_type:
        Variable type as a string, e.g. continuous, string, etc.
    series:
        Series to create the variable from. Series is None by default and in
        this case the value is ignored. If it is not supplied, then the
        variable cannot be fit.
    name:
        Name of the variable/column.
    distribution:
        Distribution to draw random values from. Can also be set by using the
        fit method.
    prop_missing:
        Proportion of the series that are missing/NA.
    dtype:
        Type of the original values, e.g. int64, float, etc. Used for type-casting
        back.
    description:
        User provided description of the variable.
    """

    dtype = "unknown"

    def __init__(self,  # pylint: disable=too-many-arguments
                 var_type: str,
                 series: Optional[Union[pl.Series, pd.Series]]=None,
                 name: Optional[str]=None,
                 distribution: Optional[BaseDistribution]=None,
                 prop_missing: Optional[float]=None,
                 dtype: Optional[str]=None,
                 description: Optional[str]=None):
        self.var_type = var_type
        self.prop_missing = prop_missing
        if series is None:
            self.name = name
            if dtype is not None:
                self.dtype = dtype
        else:
            series = _to_polars(series)
            self.name = series.name
            if prop_missing is None:
                self.prop_missing = (len(series) - len(series.drop_nulls()))/len(series)
            self.dtype = str(series.dtype)

        self.series = series
        self.distribution = distribution
        self.description = description

        if self.prop_missing is None:
            raise ValueError(f"Error while initializing variable {self.name}."
                             " prop_missing is None.")

    @classmethod
    def detect(cls, series_or_dataframe: Union[pd.Series, pl.Series, pl.DataFrame],
               description: Optional[str]=None, prop_missing: Optional[float]=None):
        """Detect variable class(es) of series or dataframe.

        Parameters
        ----------
        series_or_dataframe: pd.Series or pd.Dataframe
            If the variable is a pandas Series, then find the correct
            variable type and create an instance of that variable.
            If a Dataframe is supplied instead, a list of of variables is
            returned: one for each column in the dataframe.
        description:
            User description of the variable.
        prop_missing:
            Proportion of the values missing. If None, detect it from the series.
            Otherwise prop_missing should be a float between 0 and 1.

        Returns
        -------
        MetaVar:
            It returns a meta data variable of the correct type.
        """
        if isinstance(series_or_dataframe, (pl.DataFrame, pd.DataFrame)):
            if isinstance(series_or_dataframe, pd.DataFrame):
                return [MetaVar.detect(series_or_dataframe[col])
                        for col in series_or_dataframe]
            return [MetaVar.detect(series) for series in series_or_dataframe]

        series = _to_polars(series_or_dataframe)

        try:
            var_type = pl.datatypes.dtype_to_py_type(series.dtype).__name__
        except NotImplementedError:
            var_type = pl.datatypes.dtype_to_ffiname(series.dtype)
        return cls(cls.get_var_type(var_type), series,
                   description=description, prop_missing=prop_missing)

    @staticmethod
    def get_var_type(polars_dtype: str) -> str:
        """Convert polars dtype to MetaSynth variable type."""
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
            raise ValueError(f"Unsupported polars type '{polars_dtype}") from exc

    def to_dict(self) -> Dict[str, Any]:
        """Create a dictionary from the variable."""
        if self.distribution is None:
            dist_dict = {}
        else:
            dist_dict = self.distribution.to_dict()
        var_dict = {
            "name": self.name,
            "type": self.var_type,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": dist_dict,
        }
        if self.description is not None:
            var_dict["description"] = self.description
        return var_dict

    def __str__(self) -> str:
        """Create a readable string from a variable."""
        return str({
            "name": self.name,
            "description": self.description,
            "type": self.var_type,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": str(self.distribution),
        })

    def fit(self,
            dist: Optional[Union[str, BaseDistribution, type]]=None,
            distribution_tree: Union[str, type, BaseDistributionTree]="builtin",
            unique: Optional[bool]=None, **fit_kwargs):
        """Fit distributions to the data.

        If multiple distributions are available for the current data type,
        use the one that fits the data the best.

        While it has no arguments or return values, it will set the
        distribution attribute to the most suitable distribution.

        Parameters
        ----------
        dist:
            The distribution to fit. In case of a string, search for it
            using the aliases of all distributions. Otherwise use the
            supplied distribution (class). Examples of allowed strings are:
            "normal", "uniform", "faker.city.nl_NL". If not supplied, fit
            the best available distribution for the variable type.
        distribution_tree:
            Distribution tree to be used.
            By default use all distributions in metasynth.distribution.
        unique:
            Whether the variable should be unique. If not supplied, it will be
            inferred from the data.
        """
        if self.series is None:
            raise ValueError("Cannot fit distribution if we don't have the"
                             "original data.")

        # Automatic detection of the distribution
        disttree = get_disttree(distribution_tree)

        # Manually supplied distribution
        if dist is None:
            self.distribution = disttree.fit(self.series, self.var_type, unique=unique,
                                             **fit_kwargs)
        else:
            self.distribution = disttree.fit_distribution(dist, self.series, **fit_kwargs)

    def draw(self) -> Any:
        """Draw a random item for the variable in whatever type is required."""
        if self.distribution is None:
            raise ValueError("Cannot draw without distribution")

        # Return NA's -> None
        if self.prop_missing is not None and np.random.rand() < self.prop_missing:
            return None
        return self.distribution.draw()

    def draw_series(self, n: int) -> pl.Series:
        """Draw a new synthetic series from the metadata.

        Parameters
        ----------
        n:
            Length of the series to be created.

        Returns
        -------
        pandas.Series:
            Pandas series with the synthetic data.
        """
        if not isinstance(self.distribution, BaseDistribution):
            raise ValueError("Cannot draw without distribution.")
        self.distribution.draw_reset()
        value_list = [self.draw() for _ in range(n)]
        if "Categorical" in self.dtype:
            return pl.Series(value_list, dtype=pl.Categorical)
        return pl.Series(value_list)

    @classmethod
    def from_dict(cls, var_dict: Dict[str, Any]) -> MetaVar:
        """Restore variable from dictionary.

        Parameters
        ----------
        var_dict:
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        disttree = get_disttree()
        dist = disttree.from_dict(var_dict)
        return cls(
            var_dict["type"],
            name=var_dict["name"],
            distribution=dist,
            prop_missing=var_dict["prop_missing"], dtype=var_dict["dtype"])
