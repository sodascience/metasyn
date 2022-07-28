"""Conversion of pandas dataframes to MetaSynth datasets."""   # pylint: disable=invalid-name

from __future__ import annotations

from datetime import datetime
from importlib.resources import read_text
import json
import pathlib
from typing import Union, List, Dict, Any

import numpy as np
import pandas as pd
import jsonschema

from metasynth.var import MetaVar
from metasynth.distribution.util import _get_all_distributions
from metasynth._version import get_versions
from metasynth.distribution.base import BaseDistribution


class MetaDataset():
    """MetaSynth dataset consisting of variables.

    The MetaSynth dataset structure that is most easily created from
    a pandas dataset with the from_dataframe class method.

    Parameters
    ----------
    meta_vars:
        List of variables representing columns in a dataframe.
    n_rows:
        Number of rows in the original dataframe.
    privacy_package:
        Package that supplies the distributions.
    """

    def __init__(self, meta_vars: List[MetaVar], n_rows: int=None,
                 privacy_package: str="metasynth.distribution"):
        self.meta_vars = meta_vars
        self.n_rows = n_rows
        self.privacy_package = privacy_package

    @property
    def n_columns(self) -> int:
        """int: Number of columns of the original dataframe."""
        return len(self.meta_vars)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       distribution: dict[str, Union[str, BaseDistribution, type]]=None,
                       unique: dict[str, bool]=None,
                       privacy_package: str=None):
        """Create dataset from a Pandas dataframe.

        The pandas dataframe should be formatted already with the correct
        datatypes.

        Parameters
        ----------
        df:
            Pandas dataframe with the correct column dtypes.
        distribution:
            A dictionary that has keys that are column names and values that
            denote distributions, either with a string that gives one of their
            aliases. Or an actually fitted BaseDistribution. For example:
            {"var1": "NormalDistribution", "var2": NormalDistribution,
            "var3": NormalDistribution(0, 1)}, which are all ways to set a variable
            to a normal distribution. Note that the first two do not set the parameters
            of the distribution, while the last does.
        unique:
            A dictionary that allows specific columns to be set to be unique.
            This is only available for the integer and string datatypes. The parameter
            is ignored when the distribution is set manually. For example:
            {"var1": True, "var2": False}, which sets the first variable always to be unique,
            while it ensures that for var2, the distribution is not chosen to be unique
            (obviously while synthesizing they may still by chance be unique).
        privacy_package:
            Package that contains the implementations of the distributions.

        Returns
        -------
        MetaDataset:
            Initialized MetaSynth dataset.
        """
        if privacy_package is None:
            privacy_package = "metasynth.distribution"
        elif privacy_package == "cbs":
            privacy_package = "metasynth.privacy.cbs"

        distribution_tree = _get_all_distributions(privacy_package)

        if distribution is None:
            distribution = {}

        if unique is None:
            unique = {}

        all_vars = []
        for col_name in list(df):
            series = df[col_name]
            dist = distribution.get(col_name, None)
            unq = unique.get(col_name, None)
            var = MetaVar.detect(series)
            if dist is None:
                var.fit(distribution_tree=distribution_tree, unique=unq)
            else:
                var.fit(dist=dist, unique=unq)

            all_vars.append(var)

        return cls(all_vars, len(df), privacy_package=privacy_package)

    def to_dict(self) -> Dict[str, Any]:
        """Create dictionary with the properties for recreation."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "provenance": {
                "created by": {
                    "name": "MetaSynth",
                    "version": get_versions()["version"],
                    "privacy": self.privacy_package,
                },
                "creation time": datetime.now().isoformat()
            },
            "vars": [var.to_dict() for var in self.meta_vars],
        }

    def __getitem__(self, key: Union[int, str]) -> MetaVar:
        """Return meta var either by variable name or index."""
        if isinstance(key, int):
            # If the key is an integer, give the ith variable.
            return self.meta_vars[key]
        if isinstance(key, str):
            # If the key is a string, return the first variable with that name.
            for var in self.meta_vars:
                if var.name == key:
                    return var
            raise KeyError(f"Cannot find variable '{key}'")
        raise TypeError(f"Cannot get item for key '{key}'")

    def __str__(self) -> str:
        """Create a readable string that shows the variables."""
        cur_str = "# Rows: "+str(self.n_rows)+"\n"
        cur_str += "# Columns: "+str(self.n_columns)+"\n"
        for var in self.meta_vars:
            cur_str += "\n"+str(var)+"\n"
        return cur_str

    def to_json(self, fp: Union[pathlib.Path, str], validate: bool=True) -> None:
        """Write the MetaSynth dataset to a JSON file.

        Optional validation against a JSON schema included in the package.

        Parameters
        ----------
        fp:
            File to write the dataset to.
        validate:
            Validate the JSON file with a schema.
        """
        self_dict = _jsonify(self.to_dict())
        if validate:
            schema = json.loads(read_text("metasynth.schema", "metasynth-1_0.json"))
            jsonschema.validate(instance=self_dict, schema=schema)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self_dict, f, indent=4)

    @classmethod
    def from_json(cls, fp: Union[pathlib.Path, str], validate: bool=True) -> MetaDataset:
        """Read a MetaSynth dataset from a JSON file.

        Parameters
        ----------
        fp:
            Path to read the data from.
        validate:
            Validate the JSON file with a schema.

        Returns
        -------
        MetaDataset:
            A restored metadataset from the file.
        """
        with open(fp, "r", encoding="utf-8") as f:
            self_dict = json.load(f)

        if validate:
            schema = json.loads(read_text("metasynth.schema", "metasynth-1_0.json"))
            jsonschema.validate(instance=self_dict, schema=schema)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows)

    def synthesize(self, n: int) -> pd.DataFrame:
        """Create a synthetic pandas dataframe.

        Parameters
        ----------
        n:
            Number of rows to generate.

        Returns
        -------
        pandas.DataFrame:
            Dataframe with the synthetic data.
        """
        synth_dict = {var.name: var.draw_series(n) for var in self.meta_vars}
        return pd.DataFrame(synth_dict)


def _jsonify(data):
    if isinstance(data, (list, tuple)):
        return [_jsonify(d) for d in data]
    if isinstance(data, dict):
        return {key: _jsonify(value) for key, value in data.items()}

    if isinstance(data, (np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    if isinstance(data, np.ndarray):
        return _jsonify(data.tolist())
    return data
