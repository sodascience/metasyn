"""Testing utilities for plugins."""

from typing import Optional

import polars as pl
import jsonschema
from jsonschema.exceptions import SchemaError

from metasynth.distpkg import get_dist_package
from metasynth.dataset import _jsonify


def check_dist_type(tree_name: str, var_type: Optional[str] = None, **privacy_kwargs):
    """Test a distribution tree to check correctness.

    Arguments
    ---------
    tree_name:
        Name of the distribution tree (entrypoint).
    var_type:
        Variable type to check the distributions for. If None, check all variable
        types.
    privacy_kwargs:
        Keyword arguments that are supplied to the distribution (tree).
    """
    if var_type is None:
        for cur_var_type in get_dist_package("core").all_var_types:
            check_dist_type(tree_name, cur_var_type, **privacy_kwargs)
        return

    base_tree = get_dist_package("core")
    new_tree = get_dist_package(tree_name, **privacy_kwargs)
    for new_class in new_tree.get_dist_list(var_type):
        base_class = None
        for cur_base_class in base_tree.get_dist_list(var_type):
            if issubclass(new_class, cur_base_class):
                base_class = cur_base_class
                break
        assert base_class is not None, f"Error: cannot find subclass of {new_class}"

        assert base_class.name == new_class.name
        dist = new_class.default_distribution()
        assert isinstance(dist, base_class)
        if var_type == "category":
            series = pl.Series([dist.draw() for _ in range(100)], pl.Categorical)
        else:
            series = pl.Series([dist.draw() for _ in range(100)])
        assert isinstance(new_class.fit(series, **privacy_kwargs), base_class)
        assert new_tree.fit(series, var_type).var_type == var_type


def check_distribution_validation(package_name: str):
    """Check whether the distributions in the package can be validated positively.

    Arguments
    ---------
    package_name:
        Name of the package to validate the distributions for.
    """
    package = get_dist_package(package_name)
    for dist in package.distributions:
        schema = dist.schema()
        dist_dict = dist.default_distribution().to_dict()
        try:
            jsonschema.validate(_jsonify(dist_dict), schema)
        except SchemaError as err:
            raise ValueError(f"Failed distribution validation for {dist.__name__}") from err
