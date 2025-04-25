"""File readers to read dataset and write synthetic datasets."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type, Union

import polars as pl

_AVAILABLE_FILE_HANDLERS = {}


def filereader(*args):
    """Register a dataset so that it can be found by name."""

    def _wrap(cls):
        _AVAILABLE_FILE_HANDLERS[cls.name] = cls
        return cls

    return _wrap(*args)

class BaseFileReader(ABC):
    """Abstract file reader class to derive specific implementations from.

    The implementation class should have at least two class attributes: a :code:`name`
    for the implementation and :code:`extensions`, which is a list of extensions to be
    associated with the implementation. For example :code:`[".csv", ".tsv"]`.
    """

    name = "base"
    extensions: list[str] = []

    def __init__(self, metadata: dict[str, Any], file_name: str):
        """Initialize the file reader with metadata and the original file name.

        Parameters
        ----------
        metadata:
            A dictionary containing all the information such as metadata and
            file format directives. The structure of the metadata is determined
            by the implementation of the BaseFileReader and can be empty.
        file_name:
            file name of the original dataset.
        """
        self.metadata = metadata
        self.file_name = file_name

    def to_dict(self) -> dict[str, Any]:
        """Convert the class instance to a dictionary.

        Returns
        -------
            A dictionary containing all information to reconstruct the file reader.
        """
        if self.name not in _AVAILABLE_FILE_HANDLERS:
            warnings.warn(f"Current file reader {self.name} is not available, did you forget to use"
                          f" the decorator @filereader for the class {self.__class__}?")
        if self.name == "base":
            warnings.warn(f"Class attribute for {self.__class__} should not be 'base'."
                           " Please give another name to your file reader.")
        return {
            "file_reader_name": self.name,
            "format_metadata": self.metadata,
            "file_name": self.file_name,
        }

    @abstractmethod
    def _write_synthetic(self, df: pl.DataFrame, fp: Union[Path, str]):
        """Write synthetic file, to be implemented by file reader implementations.

        Parameters
        ----------
        df:
            Polars dataframe to be written to a file.
        fp:
            Name of the file to be written to.

        """
        raise NotImplementedError("Write synthetic is not implemented for the BaseFileReader.")

    def write_synthetic(self, df: pl.DataFrame, fp: Union[None, Path, str] = None,
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
        if fp is None:
            fp = self.file_name
        if Path(fp).is_file() and not overwrite:
            raise FileExistsError(f"File '{fp}' already exists, choose a different name or write "
                                  "to a different directory.")
        elif Path(fp).is_dir():
            fp = Path(fp) / self.file_name
        self._write_synthetic(df, fp)

    @classmethod
    @abstractmethod
    def default_reader(cls, fp: Union[Path, str]):
        """Create a defeault reader with the most likely settings for writing.

        Parameters
        ----------
        fp
            File for writing to by default.

        Returns
        -------
            An instantiated file reader with default settings.
        """
        raise NotImplementedError("Default_reader method is not implemented for the base class.")

    @classmethod
    @abstractmethod
    def from_file(cls, fp: Union[Path, str]):
        """Create a file reader from a path.

        Parameters
        ----------
        fp
            Path to read the dataset from.build

        Returns
        -------
            An initialized file reader.

        """
        raise NotImplementedError("from_file method is not implemented for the base class.")


@filereader
class SavFileReader(BaseFileReader):
    """File reader for .sav and .zsav files.

    Also stores the descriptions of the columns and makes sure that F.0 columns
    are converted to integers.
    """

    name = "spss-sav"
    extensions = [".sav", ".zsav"]

    def read_dataset(self, fp: Union[Path, str]):
        """Read the dataset without the metadata."""
        df, _ = SavFileReader._get_df_metadata(fp)
        return df

    @staticmethod
    def _get_df_metadata(fp: Union[Path, str]):
        """Read the dataset including the metadata."""
        try:
            import pyreadstat
        except ImportError as err:
            raise ImportError(
                "Please install pyreadstat to use the .sav/.zsav file reader.") from err

        pandas_df, prs_metadata = pyreadstat.read_sav(fp, apply_value_formats=True)
        df = pl.DataFrame(pandas_df)
        for col in df.columns:
            col_format = prs_metadata.original_variable_types[col]
            if (col_format.startswith("F") and col_format.endswith(".0")
                    and df[col].dtype == pl.Float64):
                df = df.with_columns(pl.col(col).cast(pl.Int64))
        return df, prs_metadata

    @classmethod
    def from_file(cls, fp: Union[Path, str]):
        """Create the file reader from a .sav or .zsav file.

        Parameters
        ----------
        fp:
            File to read the dataframe and metadata from.

        Returns
        -------
        df:
            Polars dataframe with the converted columns.
        file_reader:
            An instance of the :class:`SavFileReader` with the appropriate metadata.
        """
        if Path(fp).suffix not in [".sav", ".zsav"]:
            warnings.warn(f"Trying to read file '{fp}' with extension different from .sav or .zsav")
        if Path(fp).suffix == ".zsav":
            compress = True
        else:
            compress = False
        df, prs_metadata = cls._get_df_metadata(fp)
        # df, prs_metadata = pyreadstat.read_sav(fp, apply_value_formats=True)

        file_label = "This is a synthetic dataset created by metasyn."
        if prs_metadata.file_label is not None:
            file_label += f" Original file label: {file_label}"

        metadata = {
            "column_labels": prs_metadata.column_labels,
            "variable_format": prs_metadata.original_variable_types,
            "compress": compress,
            "variable_display_width": prs_metadata.variable_display_width,
            "file_label": file_label,
            "variable_value_labels": prs_metadata.variable_value_labels,
            "variable_measure": prs_metadata.variable_measure,
        }
        return df, cls(metadata, Path(fp).name)

    def _write_synthetic(self, df: pl.DataFrame, out_fp: Union[Path, str]):
        try:
            import pyreadstat
        except ImportError as err:
            raise ImportError(
                "Please install pyreadstat to use the .sav/.zsav file handler.") from err

        if "variable_format" in self.metadata:
            for col in df.columns:
                col_format = self.metadata["variable_format"][col]
                if (col_format.startswith("F") and not col_format.endswith(".0")
                        and df[col].dtype == pl.Float64):
                    n_round = int(col_format.split(".")[-1])
                    df = df.with_columns(pl.col(col).round(n_round))

        # Workaround bug/issue in pyreadstat, datetimes should be in ns or overflow will occur.
        pd_df = df.to_pandas()
        for col in pd_df.columns:
            if df[col].dtype.base_type() == pl.Datetime:
                pd_df[col] = pd_df[col].astype('datetime64[ns]')
        pyreadstat.write_sav(pd_df, out_fp, **self.metadata)

    @classmethod
    def default_reader(cls, fp: Union[str, Path]):
        return cls({}, Path(fp).name)

@filereader
class CsvFileReader(BaseFileReader):
    """File reader to read and write CSV files."""

    name = "csv"
    extensions = [".csv", ".tsv"]

    def read_dataset(self, fp: Union[Path, str], **kwargs):
        """Read CSV file.

        Parameters
        ----------
        fp:
            File to be read with the file reader.
        kwargs:
            Extra keyword arguments to be passed to polars.

        """
        df = pl.read_csv(
            fp, try_parse_dates=True, infer_schema_length=10000,
            null_values=self.metadata["null_value"],
            ignore_errors=True,
            separator=self.metadata["separator"],
            quote_char=self.metadata["quote_char"],
            **kwargs)
        return df

    @classmethod
    def from_file(cls, fp: Union[Path, str], separator: Optional[str] = None, eol_char: str = "\n",
                  quote_char: str = '"', null_values: Union[None, str, list[str]] = None, **kwargs):
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
        file_reader:
            File reader that read the dataset.
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
        )
        pl_keywords.update(kwargs)
        df = pl.read_csv(fp, **pl_keywords)  # type: ignore

        metadata = {
            "separator": separator,
            "line_terminator": eol_char,
            "quote_char": quote_char,
            "null_value": null_values[0],
        }
        return df, cls(metadata, Path(fp).name)

    def _write_synthetic(self, df, out_fp):
        df.write_csv(out_fp, **self.metadata)

    @classmethod
    def default_reader(cls, fp: Union[Path, str]):
        return cls({
            "separator": ",",
            "line_terminator": "\n",
            "quote_char": '"',
            "null_value": "",
        }, Path(fp).name)

@filereader
class ExcelFileReader(BaseFileReader):
    """File reader/writer for Microsoft Excel files."""

    name = "excel"
    extensions = [".xlsx", ".xls", ".xlsb"]

    @classmethod
    def from_file(cls, fp: Union[Path, str], sheet_name: Optional[str] = None):
        df = pl.read_excel(source=str(fp), sheet_name=sheet_name)
        return df, cls({"sheet_name": sheet_name}, Path(fp).name)

    def _write_synthetic(self, df, out_fp):
        df.write_excel(out_fp, **self.metadata)

    @classmethod
    def default_reader(cls, fp: Union[Path, str]):
        return cls({"sheet_name": "Sheet1"}, Path(fp).name)


def file_reader_from_dict(file_format_dict: dict) -> BaseFileReader:
    """Create a file reader from a dictionary.

    Parameters
    ----------
    file_format_dict:
        Dictionary containing information to create the file reader.
    """
    for handler_name, handler in _AVAILABLE_FILE_HANDLERS.items():
        if file_format_dict["file_reader_name"] == handler_name:
            return handler(metadata=file_format_dict["format_metadata"],
                           file_name=file_format_dict["file_name"])
    raise ValueError(f"Cannot find file reader with name '{handler_name}'.")

def get_file_reader(fp: Union[Path, str]) -> tuple[pl.DataFrame, BaseFileReader]:
    """Attempt to create file reader from a dataset.

    Default options will be used to read in the file.

    Parameters
    ----------
    fp
        Filename of the dataset to be read.

    Returns
    -------
    df:
        Dataframe with the dataset.
    file_reader:
        The file reader that has created the dataframe.

    Raises
    ------
    ValueError
        When the extension is unknown.
    """
    return get_file_reader_class(fp).from_file(fp)


def get_file_reader_class(fp: Union[Path, str]) -> Type[BaseFileReader]:
    """Get the file reader class from a filename."""
    suffix = Path(fp).suffix

    for handler_name, handler in _AVAILABLE_FILE_HANDLERS.items():
        if suffix in handler.extensions:
            return handler
    raise ValueError(f"Files with extension '{suffix}' are not supported.")

def read_csv(fp: Union[Path, str], separator: Optional[str] = None, eol_char: str = "\n",
             quote_char: str = '"', null_values: Union[str, list[str], None]=None,
             **kwargs) -> tuple[pl.DataFrame, CsvFileReader]:
    r"""Create the file reader from a file.

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
        CsvFileReader instance containing information on how to write CSV files.

    """
    return CsvFileReader.from_file(fp, separator=separator, eol_char=eol_char,
                                   quote_char=quote_char, null_values=null_values, **kwargs)

def read_tsv(*args, **kwargs) -> tuple[pl.DataFrame, CsvFileReader]:
    """Alias for :func:`read_csv`."""
    return read_csv(*args, **kwargs)

def read_sav(fp: Union[Path, str]) -> tuple[pl.DataFrame, SavFileReader]:
    """Create the file reader from a .sav or .zsav file.

    Parameters
    ----------
    fp:
        File to read the dataframe and metadata from.

    Returns
    -------
    df:
        Polars dataframe with the converted columns.
    file_reader:
        An instance of the :class:`SavFileReader` with the appropriate metadata.
    """
    return SavFileReader.from_file(fp)
