"""Module defining MetaFrames, used for creating MetaFrames from DataFrames."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, no_type_check
from warnings import warn

import numpy as np
import polars as pl

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from tqdm import tqdm

from metasyn.config import MetaConfig
from metasyn.file import BaseFileInterface, file_interface_from_dict
from metasyn.privacy import BasePrivacy, get_privacy
from metasyn.util import set_global_seeds
from metasyn.validation import validate_gmf_dict
from metasyn.var import MetaVar
from metasyn.varspec import VarSpec


class MetaFrame:
    """Container for statistical metadata describing a dataset.

    This class is used to fit a MetaFrame to a Polars DataFrame, serialize and
    save the MetaFrame to a file, read a MetaFrame from a file, and create
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

    def __init__(
        self,
        meta_vars: List[MetaVar],
        n_rows: Optional[int] = None,
        file_format: Union[None, BaseFileInterface, dict[str, Any]] = None,
    ):
        self.meta_vars = meta_vars
        self.n_rows = n_rows
        self._file_format: Union[None, dict[str, Any]]
        self.file_format = file_format  # type: ignore

    @property
    def n_columns(self) -> int:
        """int: Number of columns of the original dataframe."""
        return len(self.meta_vars)

    @classmethod
    def fit_dataframe(  # noqa: PLR0912
        cls,
        df: Optional[pl.DataFrame],
        var_specs: Optional[Union[list[VarSpec]]] = None,
        dist_registries: Optional[list[str]] = None,
        privacy: Optional[Union[BasePrivacy, dict]] = None,
        n_rows: Optional[int] = None,
        progress_bar: bool = True,
        config: Optional[Union[pathlib.Path, str, MetaConfig]] = None,
        file_format: Union[dict[str, Any], BaseFileInterface, None] = None,
    ):
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
        dist_registries:
            Distribution registries to use when fitting distributions to variables.
            Can be a list of strings or unspecified. This will overwrite the defaults if they
            were specified in the varspecs (but not the specifications per column).
        privacy:
            Privacy level to use by default. This will overwrite the defaults if they were
            specified in the varspecs (but not the specifications per column).
        n_rows:
            Number of rows registered in the MetaFrame. If left at None, it will use the number
            of rows in the input dataframe.
        progress_bar:
            Whether to display a progress bar.
        config:
            A path or MetaConfig object that contains information about the variable specifications
            , defaults, etc. Variable specs in the config parameter will be overwritten by the
            var_specs parameter.

        Returns
        -------
        MetaFrame:
            Initialized metasyn metaframe.
        """
        if isinstance(var_specs, (str, pathlib.Path, MetaConfig)) and config is None:
            warn(
                "Supplying the configuration through var_specs is deprecated and will be removed"
                f" in metasyn version 2.0. Use config={var_specs} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            config = var_specs
            var_specs = None
        # Parse the var_specs into a MetaConfig instance.
        if config is None:
            meta_config = MetaConfig([], dist_registries, defaults={"privacy": privacy})
        elif isinstance(config, (pathlib.Path, str)):
            meta_config = MetaConfig.from_toml(config)
        else:
            meta_config = config

        # var_specs overrules variable specifications in the configuration (file).
        if var_specs is not None:
            meta_config.update_varspecs(var_specs)

        if dist_registries is not None:
            meta_config.dist_registries = dist_registries  # type: ignore
        if privacy is not None:
            meta_config.defaults.privacy = privacy  # type: ignore

        if df is not None and not isinstance(df, pl.DataFrame):
            if isinstance(df, (str, pathlib.Path)):
                raise ValueError("Please provide a DataFrame as input, not a string or path.")
            df = pl.DataFrame(df)
        all_vars = []
        columns = df.columns if df is not None else []
        if df is not None:
            for col_name in (pbar := tqdm(columns, disable=not progress_bar, unit="variables")):
                desc = col_name[:5] + "…" + col_name[-6:] if len(col_name) > 11 else col_name
                pbar.set_description(f"{desc:>12}")
                var_spec = meta_config.get(col_name)
                var = MetaVar.fit(
                    df[col_name],
                    var_spec.dist_spec,
                    meta_config.dist_registries,
                    var_spec.privacy,
                    var_spec.prop_missing,
                    var_spec.description,
                )
                all_vars.append(var)

        # Data free columns to be appended
        for var_spec in meta_config.iter_var(exclude=columns):
            if not var_spec.data_free:
                raise ValueError(
                    f"Column with name '{var_spec.name}' not found and not declared as data_free."
                )
            distribution = meta_config.dist_registries.create(var_spec)
            prop_missing = 0.0 if var_spec.prop_missing is None else var_spec.prop_missing
            var = MetaVar(
                var_spec.name,
                var_spec.var_type,
                distribution,
                description=var_spec.description,
                prop_missing=prop_missing,
            )
            all_vars.append(var)
        if df is None:
            if meta_config.n_rows is None:
                raise ValueError(
                    "Please provide the number of rows in the configuration, or supply a DataFrame."
                )
            return cls(all_vars, meta_config.n_rows, file_format)
        n_rows = len(df) if n_rows is None else n_rows
        return cls(all_vars, n_rows, file_format)

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
        return cls.fit_dataframe(None, config=meta_config)

    def to_dict(self) -> Dict[str, Any]:
        """Create dictionary with the properties for recreation."""
        self_dict = {
            "gmf_version": "1.1",
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "provenance": {
                "created by": {
                    "name": "metasyn",
                    "version": version("metasyn"),
                },
                "creation time": datetime.now().isoformat(),
            },
            "file_format": self.file_format,
            "vars": [var.to_dict() for var in self.meta_vars],
        }
        if self.file_format is None:
            self_dict.pop("file_format")
        return self_dict

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
            f"Column {i + 1}: {str(var)}" for i, var in enumerate(self.meta_vars)
        )
        return f"# Rows: {self.n_rows}\n# Columns: {self.n_columns}\n\n{vars_formatted}\n"

    @property
    def file_format(self) -> Optional[dict[str, Any]]:
        return self._file_format

    @file_format.setter
    def file_format(self, new_file_format: Union[None, dict[str, Any], BaseFileInterface]):
        if isinstance(new_file_format, BaseFileInterface):
            out_file_format: Optional[dict[str, Any]] = new_file_format.to_dict()
        else:
            out_file_format = new_file_format

        self._file_format = out_file_format

    @property
    def descriptions(self) -> dict[str, str]:
        """Return the descriptions of the columns."""
        return {
            var.name: var.description
            for var in self.meta_vars
            if var.name is not None and var.description is not None
        }

    @descriptions.setter
    def descriptions(self, new_descriptions: Union[dict[str, str], Sequence[str]]):
        if isinstance(new_descriptions, dict):
            for var_name, new_desc in new_descriptions.items():
                self[var_name].description = new_desc
        else:
            assert len(new_descriptions) == self.n_columns, (
                "Descriptions need to be either a dict or a "
                "sequence with the length of the number of variables."
            )
            for i_desc, new_desc in enumerate(new_descriptions):
                self[i_desc].description = new_desc

    def save(self, fp: Optional[Union[pathlib.Path, str]], validate: bool = True) -> None:
        """Serialize and save the MetaFrame to a JSON or TOML file, following the GMF format.

        Optionally, validate the saved JSON file against the JSON schema(s) included in the
        package. A TOML cannot be validated against a schema currently.

        Parameters
        ----------
        fp:
            File to write the metaframe to.
        validate:
            Validate the JSON file with a schema. If the file is a TOML file, then this will
            be ignored.
        """
        if fp is None:
            self.save_json(fp, validate)
            return
        fp_path = Path(fp)
        if fp_path.suffix == ".toml":
            self.save_toml(fp, validate)
        else:
            self.save_json(fp, validate)

    @classmethod
    def load(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> MetaFrame:
        """Read a MetaFrame from a JSON or TOML GMF file.

        Optionally, validate the saved JSON file against the JSON schema(s) included in the
        package. A TOML cannot be validated against a schema currently.

        Parameters
        ----------
        fp:
            Path to read the data from.
        validate:
            Validate the JSON file with a schema. If the file is a TOML file, then this will
            be ignored.

        Returns
        -------
        MetaFrame:
            A restored MetaFrame from the file.
        """
        fp_path = Path(fp)
        if fp_path.suffix == ".toml":
            return cls.load_toml(fp, validate)
        else:
            return cls.load_json(fp, validate)

    def save_json(self, fp: Optional[Union[pathlib.Path, str]], validate: bool = True) -> None:
        """Serialize and save the MetaFrame to a JSON file, following the GMF format.

        Optionally, validate the saved JSON file against the JSON schema(s) included in the
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

    @classmethod
    def load_json(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> MetaFrame:
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
        return cls(meta_vars, n_rows, self_dict.get("file_format"))

    def to_json(self, fp: Union[pathlib.Path, str], validate: bool = True) -> None:
        """Export, deprecated method, use Metaframe.save_json instead."""
        warn(
            "to_json method of MetaFrame is deprecated and will be removed in the future, "
            "Use MetaFrame.save_json or MetaFrame.save instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.save_json(fp, validate)

    def export(self, fp: Union[pathlib.Path, str], validate: bool = True) -> None:
        """Export, deprecated method, use Metaframe.save instead."""
        warn(
            "Export method of MetaFrame is deprecated and will be removed in the future, "
            "Use MetaFrame.save_json or MetaFrame.save instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.save_json(fp, validate)

    @classmethod
    def from_json(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> MetaFrame:
        """Import, deprecated method, use Metaframe.load_json instead."""
        warn(
            "MetaFrame.from_json is deprecated and will be removed in the future, "
            "use MetaFrame.load_json or MetaFrame.load instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.load_json(fp, validate)

    @no_type_check
    def save_toml(self, fp: Optional[Union[pathlib.Path, str]], validate: bool = True) -> None:
        try:
            import tomlkit  # noqa: PLC0415
        except ImportError:
            raise ValueError(
                "Please install tomlkit (pip install tomlkit) to enable support for saving to toml."
            )
        self_dict = _jsonify(self.to_dict())
        if validate:
            validate_gmf_dict(self_dict)

        doc = tomlkit.loads(tomlkit.dumps(self_dict))
        doc["n_rows"].comment("Number of rows")
        doc["n_columns"].comment("""Number of columns

# This is a metadata file with (limited) statistical information about each column separately in
# a dataset. No information about correlations or other relationships between columns is included.
# This file can be used to generate privacy-conscious synthetic data, which consequently has zero
# expected correlations and relationships between columns.
# For each column, the statistics can be either manually specified or estimated from real data.
# This information, including how the estimation was done, is shown in the metadata below.
#
# For more information, see https://github.com/sodascience/metasyn
""")
        for i in range(self.n_columns):
            var = self.meta_vars[i]
            doc["vars"][i].comment(f"Metadata for column with name {var.name}")
            fmi = round(self.n_rows * (1 - var.prop_missing))
            doc["vars"][i]["prop_missing"].comment(
                f"Fraction of missing values, remaining: {fmi} values"
            )
            # The below comment does not work, a tomlkit bug?
            # doc["vars"][i]["distribution"]["unique"].add(tomlkit.comment(
            # "Whether to generate unique values or not"))
            parameter_comments = []
            multi_default = (
                var.distribution.matches_name("multinoulli")
                and len(var.distribution.labels)
                == len(var.distribution.default_distribution(var_type=var.var_type).labels)
                and np.all(
                    var.distribution.labels == var.distribution.default_distribution(
                        var_type=var.var_type).labels
                )
            )
            if "parameters" in var.creation_method:
                parameters = ", ".join(var.creation_method["parameters"])
                parameter_comments.append(
                    f"The parameters {parameters} for column '{var.name}' were "
                    "manually set by the user, no data was (directly) used."
                )
            elif var.distribution.matches_name("multinoulli") and multi_default:
                parameter_comments.append(
                    "This mulinoulli distribution is the default one, no data was used."
                )
            elif "privacy" in var.creation_method:
                privacy = get_privacy(**var.creation_method["privacy"])
                parameter_comments.append(privacy.comment(var))
            if var.distribution.matches_name("multinoulli") and not multi_default:
                counts = (var.distribution.probs * (1 - var.prop_missing) * self.n_rows).round()
                parameter_comments.append(f"Counts: {counts.astype(int)}\n")
            par_comment = "\n# ".join(parameter_comments) + "\n\n"
            doc["vars"][i]["distribution"]["parameters"].add(tomlkit.comment(par_comment))
        if fp is None:
            print(tomlkit.dumps(doc))
        else:
            with open(fp, "w", encoding="utf-8") as f:
                tomlkit.dump(doc, f)

    @classmethod
    def load_toml(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> MetaFrame:
        with open(fp, "rb") as f:
            self_dict = tomllib.load(f)

        if validate:
            validate_gmf_dict(self_dict)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows, self_dict.get("file_format"))

    def synthesize(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> pl.DataFrame:
        """Create a synthetic Polars dataframe.

        Parameters
        ----------
        n:
            Number of rows to generate, if None, use number of rows in original dataframe.
        seed:
            Seed value for the internal random number generator. Set this to ensure reproducibility.
        progress_bar:
            Whether to display a progress bar.

        Returns
        -------
        polars.DataFrame:
            Dataframe with the synthetic data.
        """
        if n is None:
            if self.n_rows is None:
                raise ValueError(
                    "Cannot synthesize DataFrame, since number of rows is unknown."
                    "Please specify the number of rows to synthesize."
                )
            n = self.n_rows

        if seed is not None:
            set_global_seeds(seed)

        synth_dict = {}
        for var in (pbar := tqdm(self.meta_vars, disable=not progress_bar, unit="variables")):
            desc = var.name[:5] + "…" + var.name[-6:] if len(var.name) > 11 else var.name
            pbar.set_description(f"{desc:>12}")
            synth_dict[var.name] = var.draw_series(n, seed=None, progress_bar=progress_bar)

        return pl.DataFrame(synth_dict)

    def write_synthetic(
        self,
        file_name: Union[None, Path, str] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        file_format: Union[None, dict, BaseFileInterface] = None,
        overwrite: bool = False,
    ):
        """Write a synthetic dataset to a file.

        To write a synthetic dataset, by default it will try to create a file that has
        the same format as the original one. For example, if the separator of the CSV
        file was a comma, then it will write the synthetic data with the same separator.
        If the file format is not available (GMF files with older versions of metasyn
        or custom file interfaces), then you will have to supply your own file interface.

        Parameters
        ----------
        file_name:
            The filename to write the synthetic data to, by default None in which case
            the same filename will be used as for the original filename if available.
        n:
            Number of rows to be written for the new synthetic file, by default None
            in which case the number of rows of the original dataset will be used.
        seed:
            Set the seed for creating the synthetic dataset, by default None
        file_format:
            File format that determines how the file will be written. This is a dictionary
            that can be created by a file interface with the
            :meth:`metasyn.fileinterface.BaseFileInterface.to_dict` method. Example file interface
            classes are :class:`metasyn.fileinterface.CsvFileReader` and
            class:`metasyn.fileinterface.SavFileReader`. By default the file_format is None,
            in which case the file interface from the GMF file will be used, otherwise an error
            will be thrown.

        Raises
        ------
        ValueError:
            If the file format is None, and the MetaFrame object itself does not have a file format
            either.

        """
        if file_format is not None:
            if isinstance(file_format, BaseFileInterface):
                file_format = file_format.to_dict()
            if self.file_format is not None:
                if self.file_format["file_interface_name"] != file_format["file_interface_name"]:
                    warn(
                        "Writing the synthetic file with a different format as the original "
                        f"dataset. Original: {self.file_format['file_interface_name']}, "
                        f"Synthetic: {file_format['file_interface_name']}"
                    )
            self.file_format = file_format  # type: ignore
        if self.file_format is None:
            raise ValueError(
                "Cannot write synthetic dataset without file handler."
                " Use write_synthetic(..., file_format=your_file_handler.to_dict())"
            )
        file_handler = file_interface_from_dict(self.file_format)
        file_handler.check_filename(file_name, overwrite=overwrite)  # Check filename before synth
        syn_df = self.synthesize(n, seed)
        file_handler.write_file(syn_df, file_name, overwrite=overwrite)

    def __repr__(self) -> str:
        """Return the MetaFrame as it would be output to JSON."""
        if len(self.meta_vars) <= 3:
            var_str = ", ".join(var.__repr__() for var in self.meta_vars)
        else:
            var_str = ", ".join(var.__repr__() for var in self.meta_vars[:2])
            var_str += ", ..."
        return f"MetaFrame: size = ({self.n_rows} x {self.n_columns}) <{var_str}>"


def _jsonify(data):
    if isinstance(data, (list, tuple)):
        return [_jsonify(d) for d in data]
    if isinstance(data, dict):
        return {key: _jsonify(value) for key, value in data.items()}

    if isinstance(data, (np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    if isinstance(data, np.float32):
        return float(data)
    if isinstance(data, np.ndarray):
        return _jsonify(data.tolist())
    return data
