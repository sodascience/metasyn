"""Conversion of pandas dataframes to MetaSynth datasets."""   # pylint: disable=invalid-name

from __future__ import annotations

import json
import pathlib
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from importlib.resources import read_text
from typing import Any, Dict, List, Optional, Sequence, Union

import jsonschema
import numpy as np
import polars as pl

from metasynth.privacy import BasePrivacy, BasicPrivacy
from metasynth.provider import BaseDistributionProvider
from metasynth.validation import validate_gmf_dict
from metasynth.var import MetaVar


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

    def __init__(self, meta_vars: List[MetaVar],
                 n_rows: Optional[int] = None):
        self.meta_vars = meta_vars
        self.n_rows = n_rows

    @property
    def n_columns(self) -> int:
        """int: Number of columns of the original dataframe."""
        return len(self.meta_vars)

    @classmethod
    def from_dataframe(cls,
                       df: pl.DataFrame,
                       spec: Optional[dict[str, dict]] = None,
                       dist_providers: Union[str, list[str], BaseDistributionProvider,
                                             list[BaseDistributionProvider]] = "builtin",
                       privacy: Optional[BasePrivacy] = None):
        """Create dataset from a Pandas dataframe.

        The pandas dataframe should be formatted already with the correct
        datatypes.

        Parameters
        ----------
        df:
            Pandas dataframe with the correct column dtypes.
        spec:
            Column specifications to modify the defaults. For each of the columns additional
            directives can be supplied here. There are 3 different directives currently supported:

            distribution

            Set the distribution, either with a string that gives one of their
            aliases or an actually fitted BaseDistribution. For example:
            {"distribution": "NormalDistribution"} which is the same as
            {"distribution": NormalDistribution} or
            {"distribution": NormalDistribution(0, 1)}, which are all ways to set a variable
            to a normal distribution. Note that the first two do not set the parameters
            of the distribution, while the last does.

            unique

            To set a column to be unique/key.
            This is only available for the integer and string datatypes. Setting a variable
            to unique ensures that the synthetic values generated for this variable are unique.
            This is useful for ID or primary key variables, for example. The parameter...
            is ignored when the distribution is set manually. For example:
            {"unique": True}, which sets the variable to be unique or {"unique": False} which
            forces the variable to be not unique. If the uniqueness is not specified, it is
            assumed to be not unique, but might give a warning if it is detected that it might
            be.

            description

            Set the description of a variable: {"description": "Some description."}

            privacy

            Set the privacy level for a variable: {"privacy": DifferentialPrivacy(epsilon=10)}

            prop_missing

            Proportion of missing values for a variable: {"prop_missing": 0.3}

            Any number of the above directives may be set for any number of variables.
        dist_providers:
            Distribution providers to use when fitting distributions to variables.
            Can be a string, provider, or provider type.
        privacy:
            Privacy level to use by default.

        Returns
        -------
        MetaDataset:
            Initialized MetaSynth dataset.
        """
        if privacy is None:
            privacy = BasicPrivacy()
        if spec is None:
            spec = {}
        else:
            spec = deepcopy(spec)

        all_vars = []
        for col_name in df.columns:
            series = df[col_name]
            col_spec = spec.get(col_name, {})
            dist = col_spec.pop("distribution", None)
            unq = col_spec.pop("unique", None)
            description = col_spec.pop("description", None)
            prop_missing = col_spec.pop("prop_missing", None)
            cur_privacy = col_spec.pop("privacy", privacy)
            fit_kwargs = col_spec.pop("fit_kwargs", {})
            if len(col_spec) != 0:
                raise ValueError(f"Unknown spec items '{col_spec}' for variable '{col_name}'.")
            var = MetaVar.detect(series, description=description, prop_missing=prop_missing)
            var.fit(dist=dist, dist_providers=dist_providers, unique=unq, privacy=cur_privacy,
                    fit_kwargs=fit_kwargs)

            all_vars.append(var)

        return cls(all_vars, len(df))

    def to_dict(self) -> Dict[str, Any]:
        """Create dictionary with the properties for recreation."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "provenance": {
                "created by": {
                    "name": "MetaSynth",
                    "version": version("metasynth"),
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

    @property
    def descriptions(self) -> dict[str, str]:
        """Return the descriptions of the columns."""
        return {var.name: var.description for var in self.meta_vars
                if var.name is not None and var.description is not None}

    @descriptions.setter
    def descriptions(self, new_descriptions: Union[dict[str, str], Sequence[str]]):
        if isinstance(new_descriptions, dict):
            for var_name, new_desc in new_descriptions.items():
                self[var_name].description = new_desc
        else:
            assert len(new_descriptions) == self.n_columns, (
                "Descriptions need to be either a dict or a "
                "sequence with the length of the number of variables.")
            for i_desc, new_desc in enumerate(new_descriptions):
                self[i_desc].description = new_desc

    def to_json(self, fp: Union[pathlib.Path, str], validate: bool = True) -> None:
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
            validate_gmf_dict(self_dict)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self_dict, f, indent=4)

    @classmethod
    def from_json(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> MetaDataset:
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
            schema = json.loads(read_text("metasynth.schema", "generative_metadata_format.json"))
            jsonschema.validate(instance=self_dict, schema=schema)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows)

    def synthesize(self, n: int) -> pl.DataFrame:
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
        return pl.DataFrame(synth_dict)


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
