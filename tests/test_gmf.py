"""Test old GMF files."""

from pathlib import Path

import polars as pl
from pytest import mark

from metasyn import MetaFrame


@mark.parametrize("gmf_file", list(Path("tests", "data", "gmf_files").glob("*")))
def test_old_gmf_files(gmf_file):
    mf = MetaFrame.load_json(gmf_file)
    assert isinstance(mf.synthesize(10), pl.DataFrame)
