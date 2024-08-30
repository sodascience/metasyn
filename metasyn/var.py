"""Module defining the MetaVar class, which represents a metadata variable."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import polars as pl

from metasyn.distribution.base import BaseDistribution
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.provider import BaseDistributionProvider, DistributionProviderList
from metasyn.varspec import DistributionSpec


class MetaVar:
    """Metadata variable describing a column in a MetaFrame.

    MetaVar is a structure that holds all metadata needed to generate a
    synthetic column for it. This is the variable level building block for the
    MetaFrame. It contains the methods to convert a polars Series into a
    variable with an appropriate distribution. The MetaVar class is to the
    MetaFrame what a polars Series is to a DataFrame.

    This class is considered a passthrough class used by the MetaFrame class,
    and is not intended to be used directly by the user.

    Parameters
    ----------
    var_type:
        String containing the variable type, e.g. continuous, string, etc.
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
        back. The default value is "unknown".
    description:
        User-provided description of the variable.
    creation_method:
        A dictionary that contains information on how the variable was created. If None,
        it will be assumed to have been created by the user.
    """

    def __init__(
        self,  # pylint: disable=too-many-arguments
        name: str,
        var_type: str,
        distribution: BaseDistribution,
        dtype: str = "unknown",
        description: Optional[str] = None,
        prop_missing: float = 0.0,
        creation_method: Optional[dict] = None,
    ):
        self.name = name
        self.var_type = var_type
        self.distribution = distribution
        self.dtype = dtype
        self.description = description
        self.prop_missing = prop_missing
        self.creation_method = creation_method
        if creation_method is None:
            self.creation_method = {"created_by": "user"}
        if self.prop_missing < -1e-8 or self.prop_missing > 1 + 1e-8:
            raise ValueError(
                f"Cannot create variable '{self.name}' with proportion missing "
                "outside range [0, 1]"
            )

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
        if not isinstance(series, pl.Series):
            series = pl.Series(series)
        if series.dtype.base_type() in [pl.Categorical, pl.Enum]:
            polars_dtype = "categorical"
        else:
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
            "categorical": "categorical",
            "bool": "categorical",
            "NoneType": "continuous",
        }
        try:
            return convert_dict[polars_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported polars type '{polars_dtype}'") from exc

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
            "creation_method": self.creation_method,
        }
        if self.description is not None:
            var_dict["description"] = self.description
        return var_dict

    def __str__(self) -> str:
        """Return an easy to read formatted string for the variable."""
        description = f'Description: "{self.description}"\n' if self.description else ""

        if self.distribution is None:
            distribution_formatted = "No distribution information available"
        else:
            distribution_formatted = "\n".join(
                "\t" + line for line in str(self.distribution).split("\n")
            )

        return (
            f'"{self.name}"\n'
            f"{description}"
            f"- Variable Type: {self.var_type}\n"
            f"- Data Type: {self.dtype}\n"
            f"- Proportion of Missing Values: {self.prop_missing:.4f}\n"
            f"- Distribution:\n{distribution_formatted}\n"
        )

    @classmethod
    def fit(
        cls,  # pylint: disable=too-many-arguments
        series: pl.Series,
        dist_spec: Optional[Union[dict, type, BaseDistribution, DistributionSpec]] = None,
        provider_list: DistributionProviderList = DistributionProviderList("builtin"),
        privacy: BasePrivacy = BasicPrivacy(),
        prop_missing: Optional[float] = None,
        description: Optional[str] = None,
    ) -> MetaVar:
        """Fit distributions to the data.

        If multiple distributions are available for the current data type,
        use the one that fits the data the best.

        While it has no arguments or return values, it will set the
        distribution attribute to the most suitable distribution.

        Parameters
        ----------
        series:
            Data series to fit a distribution to.
        dist_spec:
            The distribution to fit. In case of a string, search for it
            using the aliases of all distributions. Otherwise use the
            supplied distribution (class). Examples of allowed strings are:
            "normal", "uniform", "faker.city.nl_NL". If not supplied, fit
            the best available distribution for the variable type.
        provider_list:
            Distribution providers that are used for fitting.
        privacy:
            Privacy level to use for fitting the series.
        prop_missing:
            Proportion of the values missing, default None.
        description:
            Description for the variable.
        """
        if not isinstance(series, pl.Series):
            series = pl.Series(series)
        var_type = cls.get_var_type(series)
        dist_spec = DistributionSpec.parse(dist_spec)
        distribution = provider_list.fit(series, var_type, dist_spec, privacy)
        if prop_missing is None:
            prop_missing = (len(series) - len(series.drop_nulls())) / len(series)
        return cls(
            series.name,
            var_type,
            distribution=distribution,
            dtype=str(series.dtype),
            description=description,
            prop_missing=prop_missing,
            creation_method=dist_spec.get_creation_method(privacy),
        )

    def draw(self) -> Any:
        """Draw a random item for the variable in whatever type is required."""
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
        polars.Series:
            Polars series with the synthetic data.
        """
        self.distribution.draw_reset()
        value_list = [self.draw() for _ in range(n)]
        if "Categorical" in self.dtype:
            return pl.Series(value_list, dtype=pl.Categorical)
        return pl.Series(value_list)

    @classmethod
    def from_dict(
        cls,
        var_dict: Dict[str, Any],
        distribution_providers: Union[
            None, str, type[BaseDistributionProvider], BaseDistributionProvider
        ] = None,
    ) -> MetaVar:
        """Restore variable from dictionary.

        Parameters
        ----------
        distribution_providers:
            Distribution providers to use to create the variable. If None,
            use all installed/available distribution providers.
        var_dict:
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        provider_list = DistributionProviderList(distribution_providers)
        dist = provider_list.from_dict(var_dict)
        return cls(
            name=var_dict["name"],
            var_type=var_dict["type"],
            distribution=dist,
            prop_missing=var_dict["prop_missing"],
            dtype=var_dict["dtype"],
            description=var_dict.get("description", None),
            creation_method=var_dict.get("creation_method", {"created_by": "unknown"}),
        )
