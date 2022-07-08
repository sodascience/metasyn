"""Module with utilities that supports the distributions."""

import inspect
import importlib
import pkgutil
try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from metasynth.distribution.base import BaseDistribution
from metasynth.distribution import ContinuousDistribution
from metasynth.distribution import DiscreteDistribution
from metasynth.distribution import StringDistribution
from metasynth.distribution import CategoricalDistribution

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


def _get_all_distributions(pkg_name):
    distributions = {
        "discrete": [],
        "continuous": [],
        "string": [],
        "categorical": [],
    }
    pkg_path = files(pkg_name)
    modules = [x for x in pkgutil.walk_packages(path=[str(pkg_path)], prefix=pkg_name + ".")
               if not x.ispkg]
    for _, mod_name, _ in modules:
        cur_module = importlib.import_module(mod_name)
        mod_classes = inspect.getmembers(
            cur_module, inspect.isclass)
        for _name, dist in mod_classes:
            if dist.__module__ != mod_name:
                continue
            if issubclass(dist, DiscreteDistribution):
                distributions["discrete"].append(dist)
            elif issubclass(dist, ContinuousDistribution):
                distributions["continuous"].append(dist)
            elif issubclass(dist, StringDistribution):
                distributions["string"].append(dist)
            elif issubclass(dist, CategoricalDistribution):
                distributions["categorical"].append(dist)
    return distributions
