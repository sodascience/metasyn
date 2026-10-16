"""Utility module for metasyn."""
from __future__ import annotations

import random
from collections import defaultdict

import faker
import numpy as np
import polars as pl

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from importlib.resources import files

ALL_VAR_TYPES = ["discrete", "continuous", "time", "date", "datetime", "string", "categorical",
                 "duration"]

CONVERT_DICT = {
    "UInt8": "discrete",
    "UInt16": "discrete",
    "UInt32": "discrete",
    "UInt64": "discrete",
    "Int8": "discrete",
    "Int16": "discrete",
    "Int32": "discrete",
    "Int64": "discrete",
    "Null": "continuous",
    "Float32": "continuous",
    "Float64": "continuous",
    "Date": "date",
    "Datetime": "datetime",
    "Time": "time",
    "Duration": "duration",
    "String": "string",
    "Categorical": "categorical",
    "Enum": "categorical",
    "Boolean": "categorical",
    "Object": "mixed",
}


def get_registry() -> dict:
    """Get the registry dictionary from the package.

    This registry contains information on plugins that are available for metasyn.

    Returns
    -------
        Dictionary containing the registry entries.
    """
    registry_fp = files(__package__) / "schema" / "plugin_registry.toml"
    with registry_fp.open("rb") as handle:
        registry = tomllib.load(handle)
    return registry

def set_global_seeds(seed: int):
    """Set the global seeds for all random number generators.

    Parameters
    ----------
    seed
        The seed to use for the random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
    faker.Faker.seed(seed)


def get_var_type(series: pl.Series) -> str:
    """Convert polars dtype to metasyn variable type.

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
    polars_dtype = str(series.dtype.base_type())
    try:
        return CONVERT_DICT[polars_dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported polars type '{polars_dtype}'") from exc

def get_dtype(var_type):
    reverse_dict = defaultdict(list)
    for dtype, vtype in CONVERT_DICT.items():
        reverse_dict[vtype].append(dtype)
    return reverse_dict[var_type][-1]
