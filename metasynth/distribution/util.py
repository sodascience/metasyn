"""Module with utilities that supports the distributions."""

import inspect
import importlib
from typing import Sequence, Tuple

import numpy as np

from metasynth.distribution.base import BaseDistribution

SIDE_LEFT = -1
SIDE_RIGHT = -2


def get_dist_class(name):
    """Obtain a distribution and fit arguments from a name

    For example, if we use "faker.city.nl_NL", we should get the FakerDistribution
    with the key word arguments/dictionary: {faker_type: city, locale: nl_NL}.

    Parameters
    ----------
    name: str
        Name of the distribution.

    Returns
    -------
    tuple of BaseDistribution, dict:
        The class that relates to the distribution name and the fit
        keyword arguments as a dictionary. Most distributions will have
        an empty dictionary.
    """
    modules = [
        "metasynth.distribution.continuous",
        "metasynth.distribution.discrete",
        "metasynth.distribution.faker",
        "metasynth.distribution.regex.base",
        "metasynth.distribution.categorical"
    ]

    # Iterate over all distribution modules
    for module_str in modules:
        module = importlib.import_module(module_str)
        for _, dist_class in inspect.getmembers(module, inspect.isclass):
            # Check if it comes originally from the current module.
            if not dist_class.__module__ == module.__name__:
                continue

            # Check if it is a distribution
            if not issubclass(dist_class, BaseDistribution):
                continue

            # Ask the distribution if the name belongs to them
            if dist_class.is_named(name):
                return dist_class, dist_class.fit_kwargs(name)
    raise ValueError(f"Cannot find distribution with name {name}")
