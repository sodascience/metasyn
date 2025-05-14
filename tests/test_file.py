from pathlib import Path

import polars as pl
import pytest
from pytest import mark

import metasyn as ms
from metasyn.demo.dataset import _AVAILABLE_DATASETS, demo_dataframe, demo_file
from metasyn.file import (
    _AVAILABLE_FILE_INTERFACES,
    BaseFileInterface,
    CsvFileInterface,
    SavFileInterface,
    StataFileInterface,
    file_interface_from_dict,
    fileinterface,
    read_file,
    read_csv,
    read_dta,
    read_sav,
    read_tsv,
)


class BadFileInterface1(BaseFileInterface):
    name = "bad1"
    def _write_file(self, df, fp):
        pass

    @classmethod
    def default_interface(cls, fp):
        return BaseFileInterface.default_interface(fp)

    @classmethod
    def read_file(cls, fp):
        return BaseFileInterface.read_file(fp)


@fileinterface
class BadFileInterface2(BaseFileInterface):
    def _write_file(self, df, fp):
        pass

    @classmethod
    def default_interface(cls, fp):
        pass

    @classmethod
    def read_file(cls, fp):
        pass

    @classmethod
    def default_interface(cls, fp):
        pass

    @classmethod
    def read_file(cls, fp):
        pass

@fileinterface
class BadFileInterface3(BaseFileInterface):
    name = "also_bad"
    def _write_file(self, df, fp):
        super()._write_file(df, fp)

    @classmethod
    def default_interface(cls, fp):
        pass

    @classmethod
    def read_file(cls, fp):
        pass

def test_decorator_warning():
    with pytest.warns(UserWarning):
        BadFileInterface1({}, "x").to_dict()
    with pytest.warns(UserWarning):
        BadFileInterface2({}, "x").to_dict()
        print(BadFileInterface2({}, "x").name)

def test_notimplemented():
    with pytest.raises(NotImplementedError):
        BadFileInterface3({}, "x").write_file(None)

def test_file_exists_error():
    with pytest.raises(FileExistsError):
        BadFileInterface1({}, "x").write_file(None, Path("tests", "data", "titanic.csv")
                                                , overwrite=False)

def test_dir_does_not_exist_error():
    with pytest.raises(FileNotFoundError):
        BadFileInterface1({}, "x").write_file(None, Path("tests", "data", "more", "directories",
                                                         "titanic.csv")
                                                , overwrite=False)

def test_default_interface_error():
    with pytest.raises(NotImplementedError):
        BadFileInterface1.default_interface("some_file")

def test_read_file_error():
    with pytest.raises(NotImplementedError):
        BadFileInterface1.read_file("some_file")

def test_sav_warning():
    with pytest.warns(UserWarning):
        SavFileInterface.read_file(Path("tests", "data", "actually_a_sav_file.csv"))


@mark.parametrize("filename",
                  ["SAQ (Item 3 Reversed).sav", "GlastonburyFestival.sav",
                   "Drug.sav", "butlerpsych.sav", "twowayexample.sav"])
def test_sav_interface(filename, tmpdir):
    sav_fp = Path("tests", "data", filename)
    df = SavFileInterface({},"x" ).read_dataset(sav_fp)
    assert isinstance(df, pl.DataFrame)
    df, sav_interface = read_sav(sav_fp)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(sav_interface, SavFileInterface)
    assert len(df) > 0
    sav_interface.write_file(df, tmpdir)
    assert Path(tmpdir, filename).is_file()

    sav_interface.metadata["compress"] = True
    zsav_file = Path(tmpdir, Path(filename).stem + ".zsav")
    sav_interface.write_file(df, zsav_file)
    new_df, new_sav_interface = SavFileInterface.read_file(zsav_file)
    assert new_sav_interface.metadata["compress"]
    assert new_df.columns == df.columns


@mark.parametrize("dataset_name",
                  _AVAILABLE_DATASETS)
def test_csv_interface(dataset_name, tmpdir):
    filename = demo_file(dataset_name)
    direct_df, _ = CsvFileInterface.read_file(filename)
    assert isinstance(direct_df, pl.DataFrame)
    df, csv_interface = read_csv(filename)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(csv_interface, CsvFileInterface)
    assert len(df) == len(direct_df) and len(df) > 0

    csv_interface.write_file(df, tmpdir)
    new_df, _ = CsvFileInterface.read_file(Path(tmpdir, Path(filename).name))
    assert isinstance(new_df, pl.DataFrame)
    assert df.columns == new_df.columns

    assert isinstance(csv_interface.read_dataset(filename), pl.DataFrame)

def test_tsv_interface(tmpdir):
    filename = Path("tests", "data", "data.tsv")

    df, tsv_interface = read_tsv(filename)
    assert tsv_interface.metadata["separator"] == "\t"
    tsv_interface.write_file(df, tmpdir)
    assert Path(tmpdir, "data.tsv").is_file()

def test_csv_null():
    filename = demo_file("hospital")
    df, csv_interface = read_csv(filename)
    assert len(df["age"].drop_nulls()) == 11
    df, csv_interface = read_csv(filename, null_values="84")
    assert len(df["age"].drop_nulls()) == 10

def test_csv_encoding(tmpdir):
    df, fmt = ms.read_csv(demo_file("test"), encoding="windows-1252")
    ms.write_csv(df, tmpdir / "temp.csv", file_format=fmt)

@mark.parametrize(
    "filename,interface_class",
    [(demo_file("hospital"), CsvFileInterface),
    (Path("tests", "data", "Drug.sav"), SavFileInterface),
    (Path("tests", "data", "data.tsv"), CsvFileInterface)]
)
def test_file_interface_from_dict(filename, interface_class):
    df, file_interface = read_file(filename)
    assert isinstance(file_interface, interface_class)
    assert isinstance(file_interface_from_dict(file_interface.to_dict()), interface_class)

def test_file_interface_errors():
    with pytest.raises(ValueError):
        file_interface_from_dict({"file_interface_name": "unknown"})

    with pytest.raises(ValueError):
        read_file(Path("tests", "test_fileinterface.py"))

@mark.parametrize("interface_class",
                  [x for x in _AVAILABLE_FILE_INTERFACES.values() if not x.__name__.startswith("Bad")])
def test_default_file_interfaces(interface_class, tmpdir):
    df = demo_dataframe("test")
    suffix = interface_class.extensions[0]
    fp = Path(tmpdir/f"test_file{suffix}")
    interface_class.default_interface(fp).write_file(df, fp)

    df_new, _ = interface_class.read_file(fp)
    assert isinstance(df_new, pl.DataFrame)
    assert df_new.shape == df.shape

    # same, but using write_* and read_*
    getattr(ms, f"write_{interface_class.name}")(df, fp, overwrite=True)
    df_new, _ = getattr(ms, f"read_{interface_class.name}")(fp)
    assert isinstance(df_new, pl.DataFrame)
    assert df_new.shape == df.shape

def test_stata(tmpdir):
    df = demo_dataframe("test")
    file_out = tmpdir / "test.dta"
    ms.write_dta(df, file_out)
    # StataFileInterface.default_interface(file_out).write_file(df, file_out)
    df_new, _ = read_dta(file_out)
    for col in df.columns:
        if col.startswith(("Int", "UInt")) and not col.endswith(("64")):
            assert df_new[col].dtype == pl.Int32  # Unfixable, since stata doesn't have Int8, etc
        elif col == "UInt64":
            assert df_new[col].dtype == pl.Int64  # Bugged
        elif col in ("Date", "Datetime"):
            assert str(df_new[col].dtype.base_type()) == "Datetime"  # Bugged
        elif col == "Categorical":
            assert df_new[col].dtype == pl.String  # Bugged
        elif col == "Boolean":
            assert df_new[col].dtype == pl.Int64  # Bugged
        elif col == "NA":
            assert df_new[col].dtype == pl.Int64  # Bugged
        else:
            assert df[col].dtype == df_new[col].dtype
