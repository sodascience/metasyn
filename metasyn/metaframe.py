"""Module defining MetaFrames, used for creating MetaFrames from DataFrames."""  # pylint: disable=invalid-name

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl
from tqdm import tqdm

from metasyn.config import MetaConfig
from metasyn.privacy import BasePrivacy
from metasyn.validation import validate_gmf_dict
from metasyn.var import MetaVar
from metasyn.varspec import VarSpec


class MetaFrame():
    """Container for statistical metadata describing a dataset.

    This class is used to fit a MetaFrame to a Polars DataFrame, serialize and
    export the MetaFrame to a file, read a MetaFrame from a file, and create
    a synthetic Polars DataFrame.

    A MetaFrame represents a metadata frame, which is a structure that holds
    statistical metadata about a dataset. The data contained in a MetaFrame
    follows the Generative Metadata Format (GMF).
    The metadata is contained in a collection of MetaVar objects,
    with each MetaVar representing a column (variable).

    A MetaFrame can easily be created using the
    ``fit_dataframe`` method, which takes a Polars DataFrame and fits a
    MetaFrame to it.

    Parameters
    ----------
    meta_vars:
        List of variables representing columns in a DataFrame.
    n_rows:
        Number of rows in the original DataFrame.
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
    def fit_dataframe(  # pylint: disable=too-many-branches
            cls,
            df: Optional[pl.DataFrame],
            var_specs: Optional[Union[list[VarSpec], pathlib.Path, str, MetaConfig]] = None,
            dist_providers: Optional[list[str]] = None,
            privacy: Optional[Union[BasePrivacy, dict]] = None,
            n_rows: Optional[int] = None,
            progress_bar: bool = True):
        """Create a metasyn object from a polars (or pandas) dataframe.

        The Polars dataframe should be formatted already with the correct
        datatypes, such as pl.Categorical (or the pandas equivalent).

        Parameters
        ----------
        df:
            Polars dataframe with the correct column dtypes.
        var_specs:
            Specifications for each column/variable. These specifications are supplied as
            a list of VarSpec instances (one for each column). Alternatively, the
            specifications can be entered as a path to a .toml file. For more information
            on this approach, see the MetaConfig class or the examples in the documentation.
            By default var_specs is None, which will use the default settings for each column.
        dist_providers:
            Distribution providers to use when fitting distributions to variables.
            Can be a string, provider, or provider type. This will overwrite the defaults if they
            were specified in the varspecs (but not the specifications per column).
        privacy:
            Privacy level to use by default. This will overwrite the defaults if they were
            specified in the varspecs (but not the specifications per column).
        n_rows:
            Number of rows registered in the MetaFrame. If left at None, it will use the number
            of rows in the input dataframe.
        progress_bar:
            Whether to create a progress bar.

        Returns
        -------
        MetaFrame:
            Initialized metasyn metaframe.
        """
        # Parse the var_specs into a MetaConfig instance.
        if isinstance(var_specs, (pathlib.Path, str)):
            meta_config = MetaConfig.from_toml(var_specs)
        elif isinstance(var_specs, MetaConfig):
            meta_config = var_specs
        elif var_specs is None:
            meta_config = MetaConfig([], dist_providers, privacy)
        else:
            meta_config = MetaConfig(var_specs, dist_providers, privacy)
        if dist_providers is not None:
            meta_config.dist_providers = dist_providers  # type: ignore
        if privacy is not None:
            meta_config.privacy = privacy  # type: ignore

        if df is not None and not isinstance(df, pl.DataFrame):
            if isinstance(df, (str, pathlib.Path)):
                raise ValueError("Please provide a DataFrame as input, not a string or path.")
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
        n_rows = len(df) if n_rows is None else n_rows
        return cls(all_vars, n_rows)

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
