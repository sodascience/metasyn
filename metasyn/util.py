"""Utility module for metasyn."""
from __future__ import annotations

import random

import faker
import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

try:
    from importlib_resources import files
except ImportError:
    from importlib.resources import files  # type: ignore


ALL_VAR_TYPES = ["discrete", "continuous", "time", "date", "datetime", "string", "categorical"]

def get_registry() -> dict:
    """Get the registry dictionary from the package.

    This registry contains information on plugins that are available for metasyn.

    Returns
    -------
        Dictionary containing the registry entries.
    """
    registry_fp = files(__package__) / "schema" / "plugin_registry.toml"
    with open(registry_fp, "rb") as handle:
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
