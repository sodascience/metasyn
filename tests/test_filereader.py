from pathlib import Path

import polars as pl
import pytest
from pytest import mark

from metasyn.demo.dataset import _AVAILABLE_DATASETS, demo_file
from metasyn.filereader import (
    BaseFileReader,
    CsvFileReader,
    SavFileReader,
    file_reader_from_dict,
    filereader,
    get_file_reader,
    read_csv,
    read_sav,
    read_tsv,
)


class BadFileReader1(BaseFileReader):
    name = "bad1"
    def _write_synthetic(self, df, fp):
        pass


@filereader
class BadFileReader2(BaseFileReader):
    def _write_synthetic(self, df, fp):
        pass

@filereader
class BadFileReader3(BaseFileReader):
    name = "also_bad"
    def _write_synthetic(self, df, fp):
        super()._write_synthetic(df, fp)

def test_decorator_warning():
    with pytest.warns(UserWarning):
        BadFileReader1({}, "x").to_dict()
    with pytest.warns(UserWarning):
        BadFileReader2({}, "x").to_dict()
        print(BadFileReader2({}, "x").name)

def test_notimplemented():
    with pytest.raises(NotImplementedError):
        BadFileReader3({}, "x").write_synthetic(None)

def test_file_exists_error():
    with pytest.raises(FileExistsError):
        BadFileReader1({}, "x").write_synthetic(None, Path("tests", "data", "titanic.csv")
                                                , overwrite=False)

def test_sav_warning():
    with pytest.warns(UserWarning):
        SavFileReader.from_file(Path("tests", "data", "actually_a_sav_file.csv"))


@mark.parametrize("filename",
                  ["SAQ (Item 3 Reversed).sav", "GlastonburyFestival.sav",
                   "Drug.sav", "butlerpsych.sav", "twowayexample.sav"])
def test_sav_reader(filename, tmpdir):
    sav_fp = Path("tests", "data", filename)
    df = SavFileReader({},"x" ).read_dataset(sav_fp)
    assert isinstance(df, pl.DataFrame)
    df, sav_reader = read_sav(sav_fp)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(sav_reader, SavFileReader)
    assert len(df) > 0
    sav_reader.write_synthetic(df, tmpdir)
    assert Path(tmpdir, filename).is_file()

    sav_reader.metadata["compress"] = True
    zsav_file = Path(tmpdir, Path(filename).stem + ".zsav")
    sav_reader.write_synthetic(df, zsav_file)
    new_df, new_sav_reader = SavFileReader.from_file(zsav_file)
    assert new_sav_reader.metadata["compress"]
    assert new_df.columns == df.columns


@mark.parametrize("dataset_name",
                  _AVAILABLE_DATASETS)
def test_csv_reader(dataset_name, tmpdir):
    filename = demo_file(dataset_name)
    direct_df, _ = CsvFileReader.from_file(filename)
    assert isinstance(direct_df, pl.DataFrame)
    df, csv_reader = read_csv(filename)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(csv_reader, CsvFileReader)
    assert len(df) == len(direct_df) and len(df) > 0

    csv_reader.write_synthetic(df, tmpdir)
    new_df, _ = CsvFileReader.from_file(Path(tmpdir, Path(filename).name))
    assert isinstance(new_df, pl.DataFrame)
    assert df.columns == new_df.columns

    assert isinstance(csv_reader.read_dataset(filename), pl.DataFrame)

def test_tsv_reader(tmpdir):
    filename = Path("tests", "data", "data.tsv")

    df, tsv_reader = read_tsv(filename)
    assert tsv_reader.metadata["separator"] == "\t"
    tsv_reader.write_synthetic(df, tmpdir)
    assert Path(tmpdir, "data.tsv").is_file()

def test_csv_null():
    filename = demo_file("hospital")
    df, csv_reader = read_csv(filename)
    assert len(df["age"].drop_nulls()) == 11
    df, csv_reader = read_csv(filename, null_values="84")
    assert len(df["age"].drop_nulls()) == 10

@mark.parametrize(
    "filename,reader_class",
    [(demo_file("hospital"), CsvFileReader),
    (Path("tests", "data", "Drug.sav"), SavFileReader),
    (Path("tests", "data", "data.tsv"), CsvFileReader)]
)
def test_file_reader_from_dict(filename, reader_class):
    df, file_reader = get_file_reader(filename)
    assert isinstance(file_reader, reader_class)
    assert isinstance(file_reader_from_dict(file_reader.to_dict()), reader_class)

def test_file_reader_errors():
    with pytest.raises(ValueError):
        file_reader_from_dict({"file_reader_name": "unknown"})

    with pytest.raises(ValueError):
        get_file_reader(Path("tests", "test_filereader.py"))
