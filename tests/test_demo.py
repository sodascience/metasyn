from pathlib import Path

import polars as pl
from pytest import mark, raises

from metasyn.demo.dataset import demo_file


@mark.parametrize("dataset", ["titanic"])
def test_demo_files(dataset):
    data_fp = demo_file(dataset)
    assert data_fp.is_file()


def test_demo_non_exist():
    with raises(ValueError):
        demo_file("non-existing-dataset")


def check_titan(output_fp):
    df = pl.read_csv(output_fp)
    assert len(df.columns) == 13
    assert "Board time" in df.columns


# def test_create_titanic(tmpdir):
#     output_fp = Path(tmpdir / "titanic.csv")
#     new_output_fp = create_titanic_demo(output_fp)
#     assert new_output_fp == output_fp
#     check_titan(output_fp)
#     new_output_fp = create_titanic_demo(output_fp)
#     assert new_output_fp == output_fp
#     check_titan(output_fp)
