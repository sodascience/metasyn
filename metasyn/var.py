"""Module defining the MetaVar class, which represents a metadata variable."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import polars as pl
from tqdm import tqdm

from metasyn.distribution.base import BaseDistribution
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.util import get_var_type, set_global_seeds
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

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        var_type: Optional[str],
        distribution: BaseDistribution,
        dtype: str = "unknown",
        description: Optional[str] = None,
        prop_missing: float = 0.0,
        creation_method: Optional[dict] = None,
    ):
        self.name = name
        if var_type is None:
            var_type = get_var_type(pl.Series([distribution.draw()]))
            distribution.draw_reset()
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
                f"Cannot create variable '{self.name}' with proportion missing outside range [0, 1]"
            )
        if self.dtype == "unknown":
            if self.var_type == "categorical":
                self.dtype = "Categorical"
            else:
                self.dtype = str(pl.Series([self.distribution.draw()]).dtype)

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

    def __repr__(self) -> str:
        return f"MetaVar <{self.name}, {self.distribution.name}>"

    @classmethod
    def fit(
        cls,  # pylint: disable=too-many-arguments
        series: pl.Series,
        dist_spec: Optional[Union[dict, type, BaseDistribution, DistributionSpec]] = None,
        dist_registry: DistributionRegistry = DistributionRegistry.parse("builtin"),
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
        dist_registry:
            Distribution registry that is used for fitting.
        privacy:
            Privacy level to use for fitting the series.
        prop_missing:
            Proportion of the values missing, default None.
        description:
            Description for the variable.
        """
        if not isinstance(series, pl.Series):
            series = pl.Series(series)
        var_type = get_var_type(series)
        dist_spec = DistributionSpec.parse(dist_spec)
        distribution, fitter = dist_registry.fit(series, var_type, dist_spec, privacy)
        if prop_missing is None:
            prop_missing = (len(series) - len(series.drop_nulls())) / len(series)
        return cls(
            series.name,
            var_type,
            distribution=distribution,
            dtype=str(series.dtype),
            description=description,
            prop_missing=prop_missing,
            creation_method=dist_spec.get_creation_method(fitter),
        )

    def draw(self) -> Any:
        """Draw a random item for the variable in whatever type is required."""
        # Return NA's -> None
        if self.prop_missing is not None and np.random.rand() < self.prop_missing:
            return None
        return self.distribution.draw()

    def draw_series(self, n: int, seed: Optional[int], progress_bar: bool = True) -> pl.Series:
        """Draw a new synthetic series from the metadata.

        Parameters
        ----------
        n:
            Length of the series to be created.
        seed:
            Seed value for the internal random number generator. Set this to ensure reproducibility.
        progress_bar:
            Whether to display a progress bar.

        Returns
        -------
        polars.Series:
            Polars series with the synthetic data.
        """
        if seed is not None:
            set_global_seeds(seed)

        self.distribution.draw_reset()

        is_not_na = np.random.rand(n) >= self.prop_missing
        n_draw: int = np.sum(is_not_na)  # type: ignore
        try:
            not_na_values = self.distribution.draw_list(n_draw)
        except NotImplementedError:
            not_na_values = [self.distribution.draw()
                             for _ in tqdm(range(n_draw), disable=not progress_bar, leave=False,
                                           desc="synthesizing")]

        # Mix the values with Nones
        cum_not_na = np.cumsum(is_not_na)
        value_list = [not_na_values[cum_not_na[i]-1] if is_not_na[i] else None for i in range(n)]
        pl_type = self.dtype.split("(")[0]

        # Workaround for polars issue with numpy 2.0
        if pl_type == "Boolean":
            value_list = [None if x is None else bool(x) for x in value_list]

        # Some dtypes have extra information, discard that
        return pl.Series(value_list, dtype=getattr(pl, pl_type))

    @classmethod
    def from_dict(
        cls,
        var_dict: Dict[str, Any],
        distribution_registries: Union[None, str, list[str]] = None,
    ) -> MetaVar:
        """Restore variable from dictionary.

        Parameters
        ----------
        distribution_registries:
            Distribution registries to use to create the variable. If None,
            use all installed/available distribution registries.
        var_dict:
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        dist_registry = DistributionRegistry.parse(distribution_registries)
        dist = dist_registry.from_dict(var_dict)
        return cls(
            name=var_dict["name"],
            var_type=var_dict["type"],
            distribution=dist,
            prop_missing=var_dict["prop_missing"],
            dtype=var_dict["dtype"],
            description=var_dict.get("description", None),
            creation_method=var_dict.get("creation_method", {"created_by": "unknown"}),
        )
