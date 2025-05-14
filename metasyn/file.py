"""File interfaces to read dataset and write synthetic datasets."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Type, Union

import polars as pl

_AVAILABLE_FILE_INTERFACES = {}


def fileinterface(*args):
    """Register a dataset so that it can be found by name."""

    def _wrap(cls):
        _AVAILABLE_FILE_INTERFACES[cls.name] = cls
        return cls

    return _wrap(*args)

class BaseFileInterface(ABC):
    """Abstract file interface class to derive specific implementations from.

    The file interface facilitates the reading and writing of dataset files. In
    particular they can ensure that the output data are exactly the same as the
    input data.

    The implementation class should have at least two class attributes: a :code:`name`
    for the implementation and :code:`extensions`, which is a list of extensions to be
    associated with the implementation. For example :code:`[".csv", ".tsv"]`.
    """

    name = "base"
    extensions: list[str] = []

    def __init__(self, metadata: dict[str, Any], file_name: str):
        """Initialize the file interface with metadata and the original file name.

        Parameters
        ----------
        metadata:
            A dictionary containing all the information such as metadata and
            file format directives. The structure of the metadata is determined
            by the implementation of the BaseFileInterface and can be empty.
        file_name:
            file name of the original dataset.
        """
        self.metadata = metadata
        self.file_name = file_name

    def to_dict(self) -> dict[str, Any]:
        """Convert the class instance to a dictionary.

        Returns
        -------
            A dictionary containing all information to reconstruct the file interface.
        """
        if self.name not in _AVAILABLE_FILE_INTERFACES:
            warnings.warn(f"Current file interface {self.name} is not available, "
                          "did you forget to use"
                          f" the decorator @fileinterface for the class {self.__class__}?")
        if self.name == "base":
            warnings.warn(f"Class attribute for {self.__class__} should not be 'base'."
                           " Please give another name to your file interface.")
        return {
            "file_interface_name": self.name,
            "format_metadata": self.metadata,
            "file_name": self.file_name,
        }

    @abstractmethod
    def _write_file(self, df: pl.DataFrame, fp: Union[Path, str]):
        """Write synthetic file, to be implemented by file interface implementations.

        Parameters
        ----------
        df:
            Polars dataframe to be written to a file.
        fp:
            Name of the file to be written to.

        """
        raise NotImplementedError("Write synthetic is not implemented for the BaseFileInterface.")

    def write_file(self, df: pl.DataFrame, fp: Union[None, Path, str] = None,
                        overwrite: bool = False):
        """Write the synthetic dataframe to a file.

        Parameters
        ----------
        df
            Dataframe to be written to a file.
        fp:
            File to write the dataframe to, by default None in which case
            the file will be the same as the original filename in the current
            working directory.
        overwrite:
            Allow overwriting of the file if it already exists, by default False.

        Raises
        ------
        FileExistsError
            If the file already exists and the overwrite argument is False.
        """
        fp = self.check_filename(fp, overwrite=overwrite)
        self._write_file(df, fp)

    def check_filename(self,  fp: Union[None, Path, str] = None,
                        overwrite: bool = False) -> Union[Path, str]:
        """Check whether the filename can be written to.

        Parameters
        ----------
        fp, optional
            File check the filename for, by default None
        overwrite, optional
            Whether overwriting is allowed, by default False

        Returns
        -------
            filename which could be either the same or different from fp.

        Raises
        ------
        FileExistsError
            If the file already exists and overwrite=False
        FileNotFoundError:
            If the parent directory of fp does not exist.
        """
        if fp is None:
            fp = self.file_name
        if Path(fp).is_file() and not overwrite:
            raise FileExistsError(f"File '{fp}' already exists, choose a different name or write "
                                  "to a different directory.")
        elif Path(fp).is_dir():
            fp = Path(fp) / self.file_name
        elif not Path(fp).parent.is_dir():
            raise FileNotFoundError(f"Parent directory does not exist for '{fp}'.")
        return fp

    @classmethod
    @abstractmethod
    def default_interface(cls, fp: Union[Path, str]):
        """Create a defeault interface with the most likely settings for writing.

        Parameters
        ----------
        fp
            File for writing to by default.

        Returns
        -------
            An instantiated file interface with default settings.
        """
        raise NotImplementedError("Default_interface method is not implemented for the base class.")

    @classmethod
    @abstractmethod
    def read_file(cls, fp: Union[Path, str]):
        """Create a file interface from a path.

        Parameters
        ----------
        fp
            Path to read the dataset from.build

        Returns
        -------
            An initialized file interface.

        """
        raise NotImplementedError("read_file method is not implemented for the base class.")


class ReadStatInterface(BaseFileInterface, ABC):
    """Abstract class to make it easier to create pyreadstat file interfaces."""

    interface = "unknown"

    def read_dataset(self, fp: Union[Path, str]):
        """Read the dataset without the metadata."""
        df, _ = SavFileInterface._get_df_metadata(fp)
        return df

    @classmethod
    def _read_data(cls, fp):
        try:
            import pyreadstat
        except ImportError as err:
            raise ImportError(
                f"Please install pyreadstat to use the {'/'.join(cls.extensions)} file interface."
                ) from err

        return getattr(pyreadstat, f"read_{cls.interface}")(fp, apply_value_formats=True)


    @classmethod
    def _get_df_metadata(cls, fp: Union[Path, str]):
        """Read the dataset including the metadata."""
        pandas_df, prs_metadata = cls._read_data(fp)
        df = pl.DataFrame(pandas_df)
        return cls._convert_with_orig_format(df, prs_metadata), prs_metadata


    @classmethod
    def read_file(cls, fp: Union[Path, str]):
        """Create the file interface from a .sav or .zsav file.

        Parameters
        ----------
        fp:
            File to read the dataframe and metadata from.

        Returns
        -------
        df:
            Polars dataframe with the converted columns.
        file_interface:
            An instance of the :class:`SavFileInterface` with the appropriate metadata.
        """
        if Path(fp).suffix not in cls.extensions:
            warnings.warn(f"Trying to read file '{fp}' with extension different from"
                          f" {'/'.join(cls.extensions)}")

        df, prs_metadata = cls._get_df_metadata(fp)
        return df, cls(cls._extract_metadata(prs_metadata, fp), Path(fp).name)

    def _write_file(self, df: pl.DataFrame, out_fp: Union[Path, str]):
        try:
            import pyreadstat
        except ImportError as err:
            raise ImportError(
                f"Please install pyreadstat to write {'/'.join(self.extensions)} files.") from err
        pd_df = self._prep_df_for_writing(df)
        label_start = "This is a synthetic dataset created by metasyn. Original label: "
        metadata = {k: v for k, v in self.metadata.items()}
        orig_file_label = metadata.pop("file_label", None)
        orig_file_label = "" if orig_file_label is None else orig_file_label
        metadata["variable_format"] = self._get_format(pd_df, df)
        if not orig_file_label.startswith(label_start):
            file_label = label_start + orig_file_label
        else:
            file_label = orig_file_label
        getattr(pyreadstat, f"write_{self.interface}")(pd_df, out_fp, **metadata,
                                                    file_label=file_label)

    @classmethod
    def default_interface(cls, fp: Union[str, Path]):
        return cls({}, Path(fp).name)

    @classmethod
    @abstractmethod
    def _convert_with_orig_format(cls, df, prs_metadata):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _extract_metadata(cls, prs_metadata, fp):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _prep_df_for_writing(cls, df):
        raise NotImplementedError()

    def _get_format(self, pd_df, df):  # noqa: ARG002
        return self.metadata.get("variable_format", {})


@fileinterface
class SavFileInterface(ReadStatInterface):
    """File interface for .sav and .zsav files.

    Also stores the descriptions of the columns and makes sure that F.0 columns
    are converted to integers.
    """

    name = "sav"
    extensions = [".sav", ".zsav"]
    interface = "sav"

    @classmethod
    def _convert_with_orig_format(cls, df, prs_metadata):
        """Read the dataset including the metadata."""
        for col in df.columns:
            col_format = prs_metadata.original_variable_types[col]
            if (col_format.startswith("F") and col_format.endswith(".0")
                    and df[col].dtype == pl.Float64):
                df = df.with_columns(pl.col(col).cast(pl.Int64))
        return df

    @classmethod
    def _extract_metadata(cls, prs_metadata, fp):
        if Path(fp).suffix == ".zsav":
            compress = True
        else:
            compress = False
        metadata = {
            "column_labels": prs_metadata.column_labels,
            "variable_format": prs_metadata.original_variable_types,
            "compress": compress,
            "variable_display_width": prs_metadata.variable_display_width,
            "file_label": prs_metadata.file_label,
            "variable_value_labels": prs_metadata.variable_value_labels,
            "variable_measure": prs_metadata.variable_measure,
        }
        return metadata


    def _prep_df_for_writing(self, df):
        if "variable_format" in self.metadata:
            for col in df.columns:
                col_format = self.metadata["variable_format"][col]
                if (col_format.startswith("F") and not col_format.endswith(".0")
                        and df[col].dtype == pl.Float64):
                    n_round = int(col_format.split(".")[-1])
                    df = df.with_columns(pl.col(col).round(n_round))
        pd_df = df.to_pandas()
        for col in pd_df.columns:
            if df[col].dtype.base_type() == pl.Datetime:
                pd_df[col] = pd_df[col].astype('datetime64[ns]')
        return pd_df

    @classmethod
    def default_interface(cls, fp: Union[str, Path]):
        return cls({}, Path(fp).name)


@fileinterface
class StataFileInterface(ReadStatInterface):
    """File interface for .dta files."""

    name = "dta"
    extensions = [".dta"]
    interface = "dta"

    @classmethod
    def _convert_with_orig_format(cls, df, prs_metadata):
        for col in df.columns:
            col_format = prs_metadata.original_variable_types[col]
            # print(col, col_format)
            if col_format == "%8.0g":
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col_format == "%12.0g":
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col_format == "%9.0g":
                df = df.with_columns(pl.col(col).cast(pl.Float32))
            elif col_format == "%10.0g":
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            # elif col_format == "%td":
                # print(col, df[col])
        # print(df["Date"])
        return df

    @classmethod
    def _extract_metadata(cls, prs_metadata, fp):  # noqa: ARG003
        metadata = {
            "column_labels": prs_metadata.column_labels,
            "variable_format": prs_metadata.original_variable_types,
            "file_label": prs_metadata.file_label,
            "variable_value_labels": prs_metadata.variable_value_labels,
        }
        return metadata

    def _prep_df_for_writing(self, df):
        pd_df = df.to_pandas()
        for col in pd_df.columns:
            if pd_df[col].dtype == "float64":
                if str(df[col].dtype).startswith(("Int", "UInt")):
                    pd_df[col] = pd_df[col].astype(str(df[col].dtype))
            if df[col].dtype.base_type() == pl.Datetime or df[col].dtype.base_type() == pl.Date:
                pd_df[col] = pd_df[col].astype('datetime64[ns]')
        return pd_df

    def _get_format(self, pd_df, df):  # noqa: ARG002
        """Fill in the formats for which we don't know them in the metadata."""
        var_format = {}
        for col in pd_df.columns:
            if col in self.metadata.get("variable_format", {}):
                continue
            pd_dtype = str(pd_df[col].dtype)
            # print(col, pd_dtype)
            if pd_dtype.startswith(("Int", "UInt")) and not pd_dtype.endswith("64"):
                var_format[col] = "%8.0g"
            elif pd_dtype.startswith(("Int", "UInt")) and pd_dtype.endswith("64"):
                var_format[col] = "%12.0g"
            elif pd_dtype == "float32":
                var_format[col] = "%9.0g"
            elif pd_dtype == "float64":
                var_format[col] = "%10.0g"
            # The below doesn't work due to overflow problems.
            # elif df[col].dtype == pl.Date:
                # var_format[col] = "%td"
                # pd_df[col] = 0

        var_format.update(self.metadata.get("variable_format", {}))
        return var_format


@fileinterface
class CsvFileInterface(BaseFileInterface):
    """File interface to read and write CSV files."""

    name = "csv"
    extensions = [".csv", ".tsv"]

    def read_dataset(self, fp: Union[Path, str], **kwargs):
        """Read CSV file.

        Parameters
        ----------
        fp:
            File to be read with the file interface.
        kwargs:
            Extra keyword arguments to be passed to polars.

        """
        df = pl.read_csv(
            fp, try_parse_dates=True, infer_schema_length=10000,
            null_values=self.metadata["null_value"],
            ignore_errors=True,
            separator=self.metadata["separator"],
            quote_char=self.metadata["quote_char"],
            encoding=self.metadata.get("encoding", "utf-8"),
            **kwargs)
        return df

    @classmethod
    def read_file(cls, fp: Union[Path, str], separator: Optional[str] = None, eol_char: str = "\n",
                  quote_char: str = '"', null_values: Union[None, str, list[str]] = None,
                  encoding="utf-8", **kwargs):
        r"""Read a csv file.

        See :func:`read_csv` for more detail.

        Parameters
        ----------
        fp:
            File to be read.
        separator:
            Separator, by default None
        eol_char:
            End of line character, by default "\\n"
        quote_char:
            Quotation character, by default '"'
        null_values:
            Null values, by default None

        Returns
        -------
        df:
            Polars dataframe for the file.
        file_interface:
            File interface that read the dataset.
        """
        if Path(fp).suffix == ".tsv" and separator is None:
            separator = "\t"
        if separator is None:
            separator = ","
        if null_values is None:
            null_values = ["", "na", "NA", "N/A", "Na"]
        if isinstance(null_values, str):
            null_values = [null_values]

        pl_keywords = dict(
            try_parse_dates=True,
            infer_schema_length=10000,
            null_values=null_values,
            ignore_errors=True,
            separator=separator,
            quote_char=quote_char,
            eol_char=eol_char,
            encoding=encoding,
        )
        pl_keywords.update(kwargs)
        df = pl.read_csv(fp, **pl_keywords)  # type: ignore

        metadata = {
            "separator": separator,
            "line_terminator": eol_char,
            "quote_char": quote_char,
            "null_value": null_values[0],
            "encoding": encoding,
        }
        return df, cls(metadata, Path(fp).name)

    def _write_file(self, df, out_fp):
        meta_copy = {k: v for k, v in self.metadata.items()}
        encoding = meta_copy.pop("encoding", "utf-8")
        if encoding == "utf-8":
            df.write_csv(out_fp, **meta_copy)
        else:
            handle = BytesIO()
            df.write_csv(handle, **meta_copy)
            handle.seek(0)
            with open(out_fp, "w", encoding=encoding) as file_handle:
                file_handle.write(handle.read().decode("utf-8"))

    @classmethod
    def default_interface(cls, fp: Union[Path, str]):
        return cls({
            "separator": ",",
            "line_terminator": "\n",
            "quote_char": '"',
            "null_value": "",
        }, Path(fp).name)

@fileinterface
class ExcelFileInterface(BaseFileInterface):
    """File interface/writer for Microsoft Excel files."""

    name = "excel"
    extensions = [".xlsx", ".xls", ".xlsb"]

    @classmethod
    def read_file(cls, fp: Union[Path, str], sheet_name: Optional[str] = None):
        df = pl.read_excel(source=str(fp), sheet_name=sheet_name)
        return df, cls({"worksheet": sheet_name}, Path(fp).name)

    def _write_file(self, df, out_fp):
        df.write_excel(out_fp, **self.metadata)

    @classmethod
    def default_interface(cls, fp: Union[Path, str]):
        return cls({"worksheet": "Sheet1"}, Path(fp).name)


def file_interface_from_dict(file_format_dict: dict) -> BaseFileInterface:
    """Create a file interface from a dictionary.

    Parameters
    ----------
    file_format_dict:
        Dictionary containing information to create the file interface.
    """
    for handler_name, handler in _AVAILABLE_FILE_INTERFACES.items():
        if file_format_dict["file_interface_name"] == handler_name:
            return handler(metadata=file_format_dict["format_metadata"],
                           file_name=file_format_dict["file_name"])
    raise ValueError(f"Cannot find file interface with name '{handler_name}'.")

def get_file_interface(fp: Union[Path, str]) -> tuple[pl.DataFrame, BaseFileInterface]:
    """Attempt to create file interface from a dataset.

    Default options will be used to read in the file.

    Parameters
    ----------
    fp
        Filename of the dataset to be read.

    Returns
    -------
    df:
        Dataframe with the dataset.
    file_interface:
        The file interface that has created the dataframe.

    Raises
    ------
    ValueError
        When the extension is unknown.
    """
    return get_file_interface_class(fp).read_file(fp)


def get_file_interface_class(fp: Union[Path, str]) -> Type[BaseFileInterface]:
    """Get the file interface class from a filename."""
    suffix = Path(fp).suffix

    for handler_name, handler in _AVAILABLE_FILE_INTERFACES.items():
        if suffix in handler.extensions:
            return handler
    raise ValueError(f"Files with extension '{suffix}' are not supported.")

def read_csv(fp: Union[Path, str], separator: Optional[str] = None, eol_char: str = "\n",
             quote_char: str = '"', null_values: Union[str, list[str], None]=None,
             **kwargs) -> tuple[pl.DataFrame, CsvFileInterface]:
    r"""Create the file interface from a file.

    This function is a wrapper around
    `polars.read_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.read_csv.html>`
    with different defaults for some of the keywords, but all keywords should be passed through.

    Parameters
    ----------
    fp:
        File to be read.
    separator
        Separator for the csv file, by default None in which case
        the separator will be a "," for .csv files and a "\\t" for
        .tsv files.
    eol_char:
        End of line character, by default "\\n"
    quote_char:
        Quotation character, by default '"'
    null_values:
        Values that will be replaced by nulls, by default None in which
        case the defaults of polars will be used ["", "na", "NA", "N/A", "Na"].
    kwargs:
        Extra keyword arguments to be passed through to polars.

    Returns
    -------
    df:
        Data frame read from the files.
    cls:
        CsvFileInterface instance containing information on how to write CSV files.

    """
    return CsvFileInterface.read_file(fp, separator=separator, eol_char=eol_char,
                                      quote_char=quote_char, null_values=null_values, **kwargs)


def _parse_fmt(file_format, fp, interface_class) -> BaseFileInterface:
    if file_format is None:
        file_format = interface_class.default_interface(fp)
    if isinstance(file_format, dict):
        file_format = interface_class(file_format, fp)
    return file_format

def write_csv(df: Union[pl.DataFrame],
              fp: Union[Path, str],
              file_format: Union[dict, BaseFileInterface, None] = None,
              overwrite: bool = False):
    """Write to a CSV file with the same file format.

    Parameters
    ----------
    df
        DataFrame to write to a file.
    fp
        File to write to.
    file_format, optional
        File format to use for writing the file, by default None meaning to use the defaults.
    overwrite, optional
        Whether to overwrite the file if it exists, by default False
    """
    file_format = _parse_fmt(file_format, fp, CsvFileInterface)
    file_format.write_file(df, fp, overwrite=overwrite)

def read_tsv(*args, **kwargs) -> tuple[pl.DataFrame, CsvFileInterface]:
    """Alias for :func:`read_csv`."""
    return read_csv(*args, **kwargs)


def write_tsv(*args, **kwargs):
    """Alias for :func:`write_csv`."""
    return write_csv(*args, **kwargs)


def read_sav(fp: Union[Path, str]) -> tuple[pl.DataFrame, SavFileInterface]:
    """Create the file interface from a .sav or .zsav file.

    Parameters
    ----------
    fp:
        File to read the dataframe and metadata from.

    Returns
    -------
    df:
        Polars dataframe with the converted columns.
    """
    return SavFileInterface.read_file(fp)


def write_sav(df: Union[pl.DataFrame],
              fp: Union[Path, str],
              file_format: Union[dict, BaseFileInterface, None] = None,
              overwrite: bool = False):
    """Write to a SAV file with the same file format.

    Parameters
    ----------
    df
        DataFrame to write to a file.
    fp
        File to write to.
    file_format, optional
        File format to use for writing the file, by default None meaning to use the defaults.
    overwrite, optional
        Whether to overwrite the file if it exists, by default False
    """
    file_format = _parse_fmt(file_format, fp, SavFileInterface)
    file_format.write_file(df, fp, overwrite=overwrite)


def read_dta(fp: Union[Path, str]) -> tuple[pl.DataFrame, StataFileInterface]:
    """Read a .dta stata file into metadata and a DataFrame.

    Parameters
    ----------
    fp
        File to be read with .dta extension.

    Returns
    -------
    df:
        Polars dataframe with the converted columns.
    """
    return StataFileInterface.read_file(fp)


def write_dta(df: Union[pl.DataFrame],
              fp: Union[Path, str],
              file_format: Union[dict, BaseFileInterface, None] = None,
              overwrite: bool = False):
    """Write to a DTA file with the same file format.

    Parameters
    ----------
    df
        DataFrame to write to a file.
    fp
        File to write to.
    file_format, optional
        File format to use for writing the file, by default None meaning to use the defaults.
    overwrite, optional
        Whether to overwrite the file if it exists, by default False
    """
    file_format = _parse_fmt(file_format, fp, StataFileInterface)
    file_format.write_file(df, fp, overwrite=overwrite)


def read_excel(fp: Union[Path, str]) -> tuple[pl.DataFrame, ExcelFileInterface]:
    """Read an excel file and create a file interface from that.

    Parameters
    ----------
    fp
        Excel file to read.

    Returns
    -------
    df:
        Polars dataframe representing the excel dataset.
    file_interface:
        An instance of the :class:`ExcelFileInterface` used for writing excel files.
    """
    return ExcelFileInterface.read_file(fp)


def write_excel(df: Union[pl.DataFrame],
                fp: Union[Path, str],
                file_format: Union[dict, BaseFileInterface, None] = None,
                overwrite: bool = False):
    """Write to a Excel file with the same file format.

    Parameters
    ----------
    df
        DataFrame to write to a file.
    fp
        File to write to.
    file_format, optional
        File format to use for writing the file, by default None meaning to use the defaults.
    overwrite, optional
        Whether to overwrite the file if it exists, by default False
    """
    file_format = _parse_fmt(file_format, fp, ExcelFileInterface)
    file_format.write_file(df, fp, overwrite=overwrite)
