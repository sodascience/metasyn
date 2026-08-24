Creating file interfaces
------------------------

.. currentmodule:: metasyn.file

File interfaces are used to read the original dataset and write the synthetic dataset.
Metasyn implements currently four file readers:

* :class:`CsvFileInterface`
* :class:`ExcelFileInterface`
* :class:`StataFileInterface`
* :class:`SavFileInterface`

To implement a new file interface, you should create a new class that is derived from the :class:`BaseFileInterface`.
To ensure that the file reader is available to metasyn, you have to decorate the class with the :func:`@fileinterface`
decorator. At a minimum, you should also implement the following methods:

* :meth:`BaseFileInterface._write_file`, used to write the synthetic file.
* :meth:`BaseFileInterface.default_interface`, default interface with default options.
* :meth:`BaseFileInterface.read_file`, used to read the input dataset.

Below is the excel file interface as an example:

.. code-block:: python

    from metasyn.file import filereader, BaseFileReader

    @fileinterface
    class ExcelFileInterface(BaseFileInterface):
        """File interface/writer for Microsoft Excel files."""

        format = "excel"
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
