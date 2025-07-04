"""Module with tests for the command line interface."""
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import polars as pl
from pytest import fixture, mark

from metasyn import MetaFrame
from metasyn.validation import validate_gmf_dict
from metasyn.file import _AVAILABLE_FILE_INTERFACES

TMP_DIR_PATH = None


@fixture(scope="module")
def tmp_dir(tmp_path_factory) -> Path:
    """Directory with a configuration, dataset and GMF file."""
    global TMP_DIR_PATH
    if TMP_DIR_PATH is None:
        # Create a temporary input file
        TMP_DIR_PATH = tmp_path_factory.mktemp("data")
        json_path = TMP_DIR_PATH / "titanic.json"
        csv_fp = Path("tests", "data", "titanic.csv")
        csv_dt = {
            "PassengerId": pl.Int64,
            "Survived": pl.Categorical,
            "Pclass": pl.Categorical,
            "Name": str,
            "Sex": pl.Categorical,
            "SibSp": pl.Categorical,
            "Parch": pl.Categorical,
            "Ticket": str,
            "Cabin": str,
            "Embarked": pl.Categorical,
            "Age": float,
            "Fare": float
        }
        data_frame = pl.read_csv(csv_fp, schema_overrides=csv_dt)[:100]
        meta_frame = MetaFrame.fit_dataframe(data_frame, var_specs=[
            {"name": "PassengerId", "distribution": {"unique": True}}])
        meta_frame.save_json(json_path)
        config_fp = TMP_DIR_PATH / "config.ini"
        with open(config_fp, "w") as handle:
            handle.write("""
[[var]]
name = "PassengerId"
distribution = {unique = true}

[[var]]
name = "Fare"
prop_missing = 0.2
distribution = {implements = "lognormal"}
""")
    return TMP_DIR_PATH


@mark.parametrize("ext", [x.extensions[0] for x in _AVAILABLE_FILE_INTERFACES.values()])
def test_cli(tmp_dir, ext):
    """A simple integration test for creating synthetic data from a GMF file."""
    # create out file path with correct extension
    out_file = tmp_dir / f"titanic{ext}"

    # create command to run in subprocess with arguments
    cmd = [
        Path(sys.executable).resolve(),   # the python executable
        Path("metasyn", "__main__.py"),   # the cli script
        "synthesize",                     # the subcommand
        "-n 25",                          # only generate 25 samples
        tmp_dir / "titanic.json",         # the input file
        "-o",
        out_file,                         # the output file
        "--seed",
        str(1234),
    ]

    # Run the cli with different extensions
    result = subprocess.run(cmd, check=False)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert out_file.is_file()
    if ext == ".csv":
        df = pl.read_csv(out_file)
        assert len(df) == 25


@mark.parametrize("config", [True, False])
def test_create_meta(tmp_dir, config):
    """CLI test on creating metadata from a csv."""
    out_file = tmp_dir / "test.json"
    cmd = [
        Path(sys.executable).resolve(),     # the python executable
        Path("metasyn", "__main__.py"),     # the cli script
        "create-meta",                      # the subcommand
        Path("tests", "data", "titanic.csv"),  # the input file
        "-o",
        out_file                            # the output file
    ]
    if config:
        cmd.extend(["--config", Path(tmp_dir) / 'config.ini'])
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_file.is_file()
    meta_frame = MetaFrame.load_json(out_file)
    assert len(meta_frame.meta_vars) == 12


def test_schema_list():
    """Test whether all plugins/schemas are listed."""
    cmd = [
        Path(sys.executable).resolve(),     # the python executable
        Path("metasyn", "__main__.py"),     # the cli script
        "schema",
        "--list"
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode == 0
    assert "builtin" in result.stdout.decode()


def test_schema_gen(tmp_dir):
    """Test whether the metadata schemas can be created."""
    titanic_json = tmp_dir / "titanic.json"
    cmd = [
        Path(sys.executable).resolve(),     # the python executable
        Path("metasyn", "__main__.py"),     # the cli script
        "schema",
        "builtin"
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode == 0
    json_schema = json.loads(result.stdout.decode())
    with open(titanic_json, "r") as handle:
        gmf_dict = json.load(handle)
    validate_gmf_dict(gmf_dict)
    jsonschema.validate(gmf_dict, json_schema)

    cmd.append("non-existent-plugin")
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode != 0


def test_datafree(tmp_dir):
    """Test generating synthetic data from only a configuration file and no real data."""
    gmf_fp = tmp_dir / "gmf_out.json"
    syn_fp = tmp_dir / "test_out.csv"
    cmd = [
        Path(sys.executable).resolve(),     # the python executable
        Path("metasyn", "__main__.py"),     # the cli script
        "create-meta",                      # the subcommand
        "--config", Path("tests", "data", "no_data_config.toml"),
        "--output", gmf_fp,              # the output file
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode == 0
    meta_frame = MetaFrame.load_json(gmf_fp)
    assert meta_frame.n_rows == 100
    assert len(meta_frame.meta_vars) == 3
    cmd2 = [
        Path(sys.executable).resolve(),     # the python executable
        Path("metasyn", "__main__.py"),     # the cli script
        "synthesize",
        gmf_fp,
        "-o",
        syn_fp
    ]
    result = subprocess.run(cmd2, check=False, capture_output=True)
    assert result.returncode == 0
    df = pl.read_csv(syn_fp)
    assert list(df.columns) == ["PassengerId", "Name", "Cabin"]
    assert len(df) == 100

def test_custom_file_interface(tmp_dir):
    config_fp = Path("examples", "config_files", "file_interface.toml")
    input_fp = Path("tests", "data", "actually_a_csv_file.sav")
    out_gmf = tmp_dir / "temp.json"
    cmd = [
        Path(sys.executable).resolve(),
        Path("metasyn", "__main__.py"),
        "create-meta",
        input_fp,
        "--config", config_fp,
        "--output", out_gmf,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    assert result.returncode == 0
    meta_frame = MetaFrame.load_json(out_gmf)
    assert len(meta_frame.meta_vars) == 12
