Creating file readers
---------------------

.. currentmodule:: metasyn.filereader

File readers are used to read the original dataset and write the synthetic dataset.
Metasyn implements currently two file readers: :class:`CsvFileReader` and :class:`SavFileReader`.
To implement a new file reader, you should create a new class that is derived from the :class:`BaseFileReader`.
To ensure that the file reader is available to metasyn, you have to decorate the class with the :func:`filereader`
decorator. At a minimum, you should also implement the :meth:`BaseFileReader._write_synthetic` method.
This methods enables metasyn to find your newly created file reader for writing to a synthetic file.


.. code-block:: python

    from metasyn.filereader import filereader, BaseFileReader

    @filereader
    class MyFileReader(BaseFileReader):
        name = "fancy_file_reader"

        @classmethod
        def from_file(cls, fp: Union[Path, str], extra_arg=...) -> tuple[pl.DataFrame, MyFileReader]:
            df = read_fancy(fp)
            # Create the file format metadata, only add metadata that is necessary for writing.
            metadata = {
                "extra_arg": extra_arg,
                "other_metadata": ...,
            }
            return df, cls(metadata, Path(fp).name)

        def _write_synthetic(self, df: pl.DataFrame, fp: Path):
            # Write the synthetic file from the polars data frame.
            write_fancy(df, self.metadata["extra_arg"], self.metadata["other_metadata"])
