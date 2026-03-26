"""The validation module contains functions to validate the serialized output of distributions.

This ensures that the Generative Metadata Format (GMF) files are interoperable and well formed.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from importlib.metadata import entry_points

import jsonschema

from metasyn.distribution.na import NADistribution
from metasyn.registry import DistributionRegistry

SCHEMA_BASE_v11 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://sodascience.github.io/generative_metadata_format/core/1.1/generative_metadata_format",  # noqa: E501
    "type": "object",
    "properties": {
        "gmf_version": {"type": "string"},
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


SCHEMA_BASE_v2 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://sodascience.github.io/generative_metadata_format/core/2.0/generative_metadata_format",  # noqa: E501
    "type": "object",
    "properties": {
        "gmf_version": {"type": "string"},

        "provenance": {
            "type": "object",
            "properties": {
                "created by": {"type": "object"},
                "creation time": {"type": "string"}
            },
            "required": ["created by", "creation time"]
        },
        "tables" : {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n_rows": {"type": "number"},
                    "n_columns": {"type": "number"},
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
        },
    },
}



class BaseGmfParser(ABC):
    def distribution_schema(self, packages) -> dict:
        defs: list[dict] = []
        for fitter in DistributionRegistry.parse(packages).fitters:
            defs.append(fitter.distribution.schema())
        defs.append(NADistribution.schema())
        return defs

    def create_schema(self, packages: list[str]) -> dict:
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
        for fitter in DistributionRegistry.parse(packages).fitters:
            defs.append(fitter.distribution.schema())
        defs.append(NADistribution.schema())

        schema = deepcopy(self.base_schema)
        schema.update({"$defs": {"all_dist_def": {"anyOf": defs}}})
        return schema

    def parse(self, gmf_dict):
        return gmf_dict

    def validate_gmf_dict(self, gmf_dict: dict):
        """Validate a JSON dictionary of a metaframe as it would be written to a GMF file.

        Make sure that you have used the _jsonify function to convert numpy arrays to
        lists, etc.

        Arguments
        ---------
        gmf_dict:
            Dictionary containing the metasyn output for a metaframe.
        """
        packages = [entry.name for entry in entry_points(group="metasyn.distribution_registry")]
        schema = self.create_schema(packages)
        jsonschema.validate(gmf_dict, schema)

class GmfV11Parser(BaseGmfParser):
    versions: list[str] = ["1.1"]
    base_schema = SCHEMA_BASE_v11

    def parse(self, gmf_dict: dict):
        new_gmf_dict = deepcopy(gmf_dict)
        n_rows = new_gmf_dict.pop("n_rows")
        n_cols = new_gmf_dict.pop("n_columns")
        vars = new_gmf_dict.pop("vars")

        new_gmf_dict["tables"] = [
            {
                "name": "single_table",
                "n_rows": n_rows,
                "n_columns": n_cols,
                "vars": vars
            }
        ]
        return new_gmf_dict

class GmfV20Parser(BaseGmfParser):
    versions: list[str] = ["2.0", "*"]
    base_schema: str = SCHEMA_BASE_v2

def _get_parser_class(gmf_dict):
    version = gmf_dict.get("gmf_version", "1.1")

    all_parsers = [GmfV11Parser, GmfV20Parser]

    best_parser_class = None
    for parser_class in all_parsers:
        if version in parser_class.versions:
            best_parser_class = parser_class
            break

    if best_parser_class is None:
        for par in all_parsers:
            if "*" in par.versions:
                best_parser_class = par
                break

        warnings.warn("Reading GMF file with unknown GMF version, update metasyn to ensure correct "
                      "reading of the GMF file.")
    return best_parser_class

def validate_gmf_dict(gmf_dict: dict):
    parser = _get_parser_class(gmf_dict)()
    parser.validate_gmf_dict(gmf_dict)

def parse_gmf_dict(gmf_dict: dict, validate: bool = True):
    parser = _get_parser_class(gmf_dict)()
    if validate:
        parser.validate_gmf_dict(gmf_dict)
    return parser.parse(gmf_dict)
