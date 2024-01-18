"""Module defining the MetaFrame class, used for the conversion of DataFrames to MetaFrames."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from metasyn.config import MetaConfig
from metasyn.privacy import BasePrivacy
from metasyn.validation import validate_gmf_dict
from metasyn.var import MetaVar


class MetaFrame():
    """Metasyn metaframe consisting of variables.

    A MetaFrame, short for metadata frame, is a structure that holds statistical metadata
    about a dataset. The data contained in a MetaFrame is in line with the
    Generative Metadata Format (GMF). It is essentially, a collection of MetaVar objects,
    each representing a column in a dataset.

    The metaframe is most easily created from a polars dataset with the from_dataframe()
    class method.

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
    def fit_dataframe(
            cls,
            df: Optional[Union[pl.DataFrame, pd.DataFrame]],
            meta_config: Optional[MetaConfig] = None,
            var_specs: Optional[list[dict]] = None,
            dist_providers: Optional[list[str]] = None,
            privacy: Optional[Union[BasePrivacy, dict]] = None,
            progress_bar: bool = True):
        """Create a metasyn object from a polars (or pandas) dataframe.

        The Polars dataframe should be formatted already with the correct
        datatypes, such as pl.Categorical (or the pandas equivalent).

        Parameters
        ----------
        df:
            Polars dataframe with the correct column dtypes.
        meta_config:
            Column specification in MetaConfig format.
        var_specs:
            Column specifications to modify the defaults. For each of the columns additional
            directives can be supplied here. There are 3 different directives currently supported:

            distribution

            Set the distribution, either with a string that gives one of their
            aliases or an actually fitted BaseDistribution. For example:
            {"distribution": "NormalDistribution"} which is the same as
            {"distribution": NormalDistribution} or
            {"distribution": NormalDistribution(0, 1)}, which are always to set a variable
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
        progress_bar:
            Whether to create a progress bar.

        Returns
        -------
        MetaFrame:
            Initialized metasyn metaframe.
        """
        if meta_config is None:
            if privacy is None:
                privacy = {"name": "none"}
            elif isinstance(privacy, BasePrivacy):
                privacy = privacy.to_dict()
            var_specs = [] if var_specs is None else var_specs
            dist_providers = dist_providers if dist_providers is not None else ["builtin"]
            meta_config = MetaConfig(var_specs, dist_providers, privacy)
        else:
            assert privacy is None

        if isinstance(df, pd.DataFrame):
            df = pl.DataFrame(df)
        all_vars = []
        columns = df.columns if df is not None else []
        if df is not None:
            for col_name in tqdm(columns, disable=not progress_bar):
                var_spec = meta_config.get(col_name)
                var = MetaVar.fit(
                    df[col_name],
                    var_spec.dist_spec,
                    meta_config.dist_providers,
                    var_spec.privacy,
                    var_spec.prop_missing,
                    var_spec.description)
                all_vars.append(var)

        # Data free columns to be appended
        for var_spec in meta_config.iter_var(exclude=columns):
            if not var_spec.data_free:
                raise ValueError(
                    f"Column with name '{var_spec.name}' not found and not declared as "
                     "data_free.")
            distribution = meta_config.dist_providers.create(var_spec)
            var = MetaVar(
                var_spec.name,
                var_spec.var_type,
                distribution,
                description=var_spec.description,
                prop_missing=var_spec.prop_missing,
            )
            all_vars.append(var)
        if df is None:
            if meta_config.n_rows is None:
                raise ValueError("Please provide the number of rows in the configuration, "
                                 "or supply a DataFrame.")
            return cls(all_vars, meta_config.n_rows)
        return cls(all_vars, len(df))

    @classmethod
    def from_config(cls, meta_config: MetaConfig) -> MetaFrame:
        """Create a MetaFrame using a configuration, but without a DataFrame.

        Parameters
        ----------
        meta_config
            Configuration to be used for creating the new MetaFrame.

        Returns
        -------
            A created MetaFrame.
        """
        return cls.fit_dataframe(None, meta_config)

    def to_dict(self) -> Dict[str, Any]:
        """Create dictionary with the properties for recreation."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "provenance": {
                "created by": {
                    "name": "metasyn",
                    "version": version("metasyn"),
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
        """Return an easy to read formatted string for the metaframe."""
        vars_formatted = "\n".join(
            f"Column {i + 1}: {str(var)}" for i,
            var in enumerate(
                self.meta_vars))
        return (
            f"# Rows: {self.n_rows}\n"
            f"# Columns: {self.n_columns}\n\n"
            f"{vars_formatted}\n"
        )

    @property
    def descriptions(self) -> dict[str, str]:
        """Return the descriptions of the columns."""
        return {var.name: var.description for var in self.meta_vars
                if var.name is not None and var.description is not None}

    @descriptions.setter
    def descriptions(
            self, new_descriptions: Union[dict[str, str], Sequence[str]]):
        if isinstance(new_descriptions, dict):
            for var_name, new_desc in new_descriptions.items():
                self[var_name].description = new_desc
        else:
            assert len(new_descriptions) == self.n_columns, (
                "Descriptions need to be either a dict or a "
                "sequence with the length of the number of variables.")
            for i_desc, new_desc in enumerate(new_descriptions):
                self[i_desc].description = new_desc

    def export(self, fp: Optional[Union[pathlib.Path, str]],
               validate: bool = True) -> None:
        """Serialize and export the MetaFrame to a JSON file, following the GMF format.

        Optionally, validate the exported JSON file against the JSON schema(s) included in the
        package.

        Parameters
        ----------
        fp:
            File to write the metaframe to.
        validate:
            Validate the JSON file with a schema.
        """
        self_dict = _jsonify(self.to_dict())
        if validate:
            validate_gmf_dict(self_dict)
        if fp is None:
            print(json.dumps(self_dict, indent=4))
        else:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(self_dict, f, indent=4)

    def to_json(self, fp: Union[pathlib.Path, str],
                validate: bool = True) -> None:
        """Serialize and export the MetaFrame to a JSON file, following the GMF format.

        This method is a wrapper and simply calls the 'export' function.

        Optionally, validate the exported JSON file against the JSON schema(s) included in the
        package.

        Parameters
        ----------
        fp:
            File to write the metaframe to.
        validate:
            Validate the JSON file with a schema.
        """
        self.export(fp, validate)

    @classmethod
    def from_json(cls, fp: Union[pathlib.Path, str],
                  validate: bool = True) -> MetaFrame:
        """Read a MetaFrame from a JSON file.

        Parameters
        ----------
        fp:
            Path to read the data from.
        validate:
            Validate the JSON file with a schema.

        Returns
        -------
        MetaFrame:
            A restored MetaFrame from the file.
        """
        with open(fp, "r", encoding="utf-8") as f:
            self_dict = json.load(f)

        if validate:
            validate_gmf_dict(self_dict)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows)

    def synthesize(self, n: Optional[int] = None) -> pl.DataFrame:
        """Create a synthetic Polars dataframe.

        Parameters
        ----------
        n:
            Number of rows to generate, if None, use number of rows in original dataframe.

        Returns
        -------
        polars.DataFrame:
            Dataframe with the synthetic data.
        """
        if n is None:
            if self.n_rows is None:
                raise ValueError("Cannot synthesize DataFrame, since number of rows is unknown."
                                 "Please specify the number of rows to synthesize.")
            n = self.n_rows
        synth_dict = {var.name: var.draw_series(n) for var in self.meta_vars}
        return pl.DataFrame(synth_dict)

    def __repr__(self) -> str:
        """Return the MetaFrame as it would be output to JSON."""
        pretty_data = _jsonify(self.to_dict())
        output = json.dumps(pretty_data, indent=4)
        return output


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
