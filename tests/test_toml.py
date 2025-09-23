"""Tests I/O for toml configuration and GMF files."""
from filecmp import cmp
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pytest import mark

from metasyn.distribution import builtin_fitters
from metasyn.metaframe import MetaFrame
from metasyn.testutils import create_input_toml, create_md_report


def test_datafree_create(tmpdir):
    """Test whether a synthetic dataset can be generated without data."""
    temp_toml = tmpdir / "test.toml"
    create_input_toml(temp_toml)
    assert cmp(temp_toml, Path("examples", "config_files", "example_all.toml"))
    mf = MetaFrame.fit_dataframe(None, config=Path(temp_toml))

    assert isinstance(mf, MetaFrame)
    assert mf.n_columns == len(builtin_fitters)

@mark.parametrize(
    "toml_input,data", [
        (Path("examples", "config_files", "example_all.toml"), None),
        # (Path("examples", "config_files", "example_config.toml"), demo_dataframe("titanic")),
        (Path("examples", "config_files", "example_datafree.toml"), None),
    ]
)
def test_toml_save_load(tmpdir, toml_input, data):
    """Test whether TOML GMF files can be saved/loaded."""
    mf = MetaFrame.fit_dataframe(data, config=toml_input)
    mf.save(tmpdir/"test.toml")
    new_mf = MetaFrame.load(tmpdir/"test.toml")
    assert mf.n_columns == new_mf.n_columns

def test_varspec_update():
    """Check whether overwriting the  varspecs with the var_specs parameter works."""
    toml_input = Path("examples", "config_files", "example_all.toml")
    var_specs = [{
        "name": "DiscreteTruncatedNormalDistribution",
        "var_type": "discrete",
        "distribution": {
            "name": "core.normal",
            "unique": False,
            "parameters": {
                "mean": 0,
                "sd": 1,
            }
        }
    }]
    mf_normal = MetaFrame.fit_dataframe(None, config=toml_input)
    mf_varspec = MetaFrame.fit_dataframe(None, var_specs=var_specs, config=toml_input)
    assert mf_normal["DiscreteTruncatedNormalDistribution"].distribution.name == "core.truncated_normal"
    assert mf_varspec["DiscreteTruncatedNormalDistribution"].distribution.name == "core.normal"

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

    with pytest.raises(ValueError):
        MetaFrame.from_config(Path("tests", "data", "unsupported_config.toml"))

    with pytest.raises(ValueError):
        MetaFrame.from_config(Path("tests", "data", "incompatible_config.toml"))
