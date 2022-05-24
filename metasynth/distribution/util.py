import inspect
import importlib

from metasynth.distribution.base import BaseDistribution


def get_dist_class(name):
    modules = [
        "metasynth.distribution.continuous",
        "metasynth.distribution.discrete",
        "metasynth.distribution.faker",
        "metasynth.distribution.string",
        "metasynth.distribution.categorical"
    ]

#     all_dist_classes = []
    for module_str in modules:
        module = importlib.import_module(module_str)
        for _, dist_class in inspect.getmembers(module, inspect.isclass):
            if dist_class.__module__ != module.__name__:
                continue
            if not dist_class.__module__ == module.__name__:
                continue
            if not issubclass(dist_class, BaseDistribution):
                continue
            if dist_class.is_named(name):
                return dist_class, dist_class.fit_kwargs(name)
    raise ValueError(f"Cannot find distribution with name {name}")
