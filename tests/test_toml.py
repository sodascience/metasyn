"""Tests I/O for toml configuration and GMF files."""
from filecmp import cmp
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pytest import mark

from metasyn.metaframe import MetaFrame
from metasyn.provider import BuiltinDistributionProvider
from metasyn.testutils import create_input_toml, create_md_report


def test_datafree_create(tmpdir):
    """Test whether a synthetic dataset can be generated without data."""
    temp_toml = tmpdir / "test.toml"
    create_input_toml(temp_toml)
    assert cmp(temp_toml, Path("examples", "config_files", "example_all.toml"))
    mf = MetaFrame.fit_dataframe(None, var_specs=Path(temp_toml))

    assert isinstance(mf, MetaFrame)
    assert mf.n_columns == len(BuiltinDistributionProvider.distributions)

@mark.parametrize(
    "toml_input,data", [
        (Path("examples", "config_files", "example_all.toml"), None),
        # (Path("examples", "config_files", "example_config.toml"), demo_dataframe("titanic")),
        (Path("examples", "config_files", "example_datafree.toml"), None),
    ]
)
def test_toml_save_load(tmpdir, toml_input, data):
    """Test whether TOML GMF files can be saved/loaded."""
    mf = MetaFrame.fit_dataframe(data, toml_input)
    mf.save(tmpdir/"test.toml")
    new_mf = MetaFrame.load(tmpdir/"test.toml")
    assert mf.n_columns == new_mf.n_columns

@mark.parametrize(
    "gmf_file", [
        Path("examples", "gmf_files", "example_gmf_simple.json"),
        Path("examples", "gmf_files", "example_gmf_titanic.json"),
    ]
)
def test_md_output(tmpdir, gmf_file):
    """Test whether the markdown report can be generated from the GMF file."""
    create_md_report(gmf_file, tmpdir/"test.md")
    assert Path(tmpdir/"test.md").is_file()
    assert isinstance(MetaFrame.load(gmf_file), MetaFrame)

def test_toml_err():
    """Test whether errors are correctly handled."""
    with pytest.raises(FileNotFoundError):
        MetaFrame.from_config("doesnotexist.toml")

    with pytest.raises(tomllib.TOMLDecodeError):
        MetaFrame.from_config(Path("tests", "data", "bad_config.toml"))
