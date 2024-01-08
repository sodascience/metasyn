"""The validation module contains functions to validate the serialized output of distributions.

This ensures that the Generative Metadata Format (GMF) files are interoperable and well formed.
"""

from __future__ import annotations

from copy import deepcopy

try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore

import jsonschema

from metasyn.distribution.na import NADistribution
from metasyn.provider import get_distribution_provider

SCHEMA_BASE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://sodascience.github.io/generative_metadata_format/core/1.0.0/generative_metadata_format",  # noqa # pylint: disable=line-too-long
    "type": "object",
    "properties": {
        "n_rows": {"type": "number"},
        "n_columns": {"type": "number"},
        "provenance": {
            "type": "object",
            "properties": {
                "created by": {"type": "object"},
                "creation time": {"type": "string"}
            },
            "required": ["created by", "creation time"]
        },
        "vars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "type": {"enum": ["discrete", "continuous", "string", "categorical", "date",
                                      "datetime", "time"]},
                    "dtype": {"type": "string"},
                    "prop_missing": {"type": "number"},
                    "distribution": {
                        "$ref": "#/$defs/all_dist_def"
                    }
                }
            },
            "required": ["name", "type", "dtype", "provenance", "prop_missing", "distribution"]
        }
    },
    "required": ["n_rows", "n_columns", "vars"],
}


def validate_gmf_dict(gmf_dict: dict):
    """Validate a JSON dictionary of a metaframe as it would be written to a GMF file.

    Make sure that you have used the _jsonify function to convert numpy arrays to
    lists, etc.

    Arguments
    ---------
    gmf_dict:
        Dictionary containing the metasyn output for a metaframe.
    """
    packages = [entry.name for entry in entry_points(group="metasyn.distribution_provider")]
    schema = create_schema(packages)
    jsonschema.validate(gmf_dict, schema)


def create_schema(packages: list[str]) -> dict:
    """Create JSON Schema to validate a GMF file.

    Arguments
    ---------
    packages:
        List of packages to create the schema with.

    Returns
    -------
    schema:
        Schema containing all the distributions in the distribution packages.
    """
    defs: list[dict] = []
    for package_name in packages:
        pkg = get_distribution_provider(package_name)
        for dist in pkg.distributions:
            defs.append(dist.schema())
    defs.append(NADistribution.schema())

    schema = deepcopy(SCHEMA_BASE)
    schema.update({"$defs": {"all_dist_def": {"anyOf": defs}}})
    return schema
