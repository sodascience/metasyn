"""Testing utilities for plugins."""

from typing import Optional

import polars as pl
import jsonschema
from jsonschema.exceptions import SchemaError

from metasynth.provider import get_distribution_provider
from metasynth.dataset import _jsonify
from metasynth.distribution.base import BaseDistribution


# def check_dist_type(tree_name: str, var_type: Optional[str] = None, **privacy_kwargs):
#     """Test a distribution tree to check correctness.
#
#     Arguments
#     ---------
#     tree_name:
#         Name of the distribution tree (entrypoint).
#     var_type:
#         Variable type to check the distributions for. If None, check all variable
#         types.
#     privacy_kwargs:
#         Keyword arguments that are supplied to the distribution (tree).
#     """
#     if var_type is None:
#         for cur_var_type in get_distribution_provider("core").all_var_types:
#             check_dist_type(tree_name, cur_var_type, **privacy_kwargs)
#         return
#
#     base_tree = get_distribution_provider("builtin")
#     new_tree = get_distribution_provider(tree_name, **privacy_kwargs)
#     for new_class in new_tree.get_dist_list(var_type):
#         base_class = None
#         for cur_base_class in base_tree.get_dist_list(var_type):
#             if issubclass(new_class, cur_base_class):
#                 base_class = cur_base_class
#                 break
#         assert base_class is not None, f"Error: cannot find subclass of {new_class}"
#
#         assert base_class.implements == new_class.implements
#         dist = new_class.default_distribution()
#         assert isinstance(dist, base_class)
#         if var_type == "category":
#             series = pl.Series([dist.draw() for _ in range(100)], pl.Categorical)
#         else:
#             series = pl.Series([dist.draw() for _ in range(100)])
#         assert isinstance(new_class.fit(series, **privacy_kwargs), base_class)
#         assert new_tree.fit(series, var_type).var_type == var_type


def check_distribution(distribution: type[BaseDistribution], privacy: str,
                       provenance: str, **privacy_kwargs):
    """Check whether the distributions in the package can be validated positively.

    Arguments
    ---------
    package_name:
        Name of the package to validate the distributions for.
    """
    # package = get_distribution_provider(package_name)
    # for dist in package.distributions:
    schema = distribution.schema()
    dist_dict = distribution.default_distribution().to_dict()
    try:
        jsonschema.validate(_jsonify(dist_dict), schema)
    except SchemaError as err:
        raise ValueError(f"Failed distribution validation for {distribution.__name__}") from err

    assert distribution.privacy == privacy
    assert len(distribution.implements.split(".")) == 2
    assert distribution.provenance == provenance
    assert distribution.var_type != "unknown"
    dist = distribution.default_distribution()
    series = pl.Series([dist.draw() for _ in range(100)])
    new_dist = distribution.fit(series, **privacy_kwargs)
    assert isinstance(new_dist, distribution)
