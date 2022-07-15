"""Module with utilities that supports the distributions."""

import inspect
import importlib
import pkgutil
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore


from metasynth.distribution.base import BaseDistribution, ScipyDistribution


SIDE_LEFT = -1
SIDE_RIGHT = -2


def get_dist_class(name, pkg_name: str="metasynth.distribution") -> Tuple[BaseDistribution, Dict]:
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
    distributions = _get_all_distributions(pkg_name)

    # Iterate over all distribution modules
    for dist_list in distributions.values():
        for dist in dist_list:
            if dist.is_named(name):
                return dist, dist.fit_kwargs(name)

    raise ValueError(f"Cannot find distribution with name {name}")


def _get_all_distributions(pkg_name: str="metasynth.distribution") -> Dict[str, List[Any]]:
    """Get all distributions from a package.

    It recursively goes through all modules and subpackages attempting to find
    any distributions derived from the BaseDistribution class.

    Arguments
    ---------
    pkg_name: Name of the (sub)package, e.g. 'metasynth.distribution'.

    Returns
    -------
    Dictionary containing lists of distributions sorted by variable type.
    """
    distributions: Dict[str, List] = defaultdict(lambda: [])#{
        # "discrete": [],
        # "continuous": [],
        # "string": [],
        # "categorical": [],
        # "time": [],
        # "date"
    # }
    # Find the package path
    pkg_path = files(pkg_name)
    modules = [x for x in pkgutil.walk_packages(path=[str(pkg_path)], prefix=pkg_name + ".")
               if not x.ispkg]

    # Iterate over all modules to find distribution.
    for _, mod_name, _ in modules:
        cur_module = importlib.import_module(mod_name)
        mod_classes = inspect.getmembers(
            cur_module, inspect.isclass)
        for _name, dist in mod_classes:
            if dist.__module__ != mod_name:
                continue
            if issubclass(dist, BaseDistribution) and not inspect.isabstract(dist) and dist != ScipyDistribution:
                distributions[dist.var_type].append(dist)
#

            # if issubclass(dist, DiscreteDistribution):
                # distributions["discrete"].append(dist)
            # elif issubclass(dist, ContinuousDistribution):
                # distributions["continuous"].append(dist)
            # elif issubclass(dist, StringDistribution):
                # distributions["string"].append(dist)
            # elif issubclass(dist, CategoricalDistribution):
                # distributions["categorical"].append(dist)
    return distributions
