import sys
import subprocess
from pathlib import Path
from pytest import mark, fixture
import polars as pl
from metasyn import MetaFrame

TMP_DIR_PATH = None

@fixture(scope="module")
def tmp_dir(tmp_path_factory) -> Path:
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
        data_frame = pl.read_csv(csv_fp, dtypes=csv_dt)[:100]
        meta_frame = MetaFrame.fit_dataframe(data_frame, spec={"PassengerId": {"unique": True}})
        meta_frame.to_json(json_path)
    return TMP_DIR_PATH

@mark.parametrize("ext", [".csv", ".feather", ".parquet", ".pkl", ".xlsx"])
def test_cli(tmp_dir, ext):
    """A simple integration test for reading and writing using the CLI"""
    
    # create out file path with correct extension
    out_file = tmp_dir / f"titanic{ext}"

    # create command to run in subprocess with arguments
    cmd = [
        Path(sys.executable).resolve(),   # the python executable
        Path("metasyn", "__main__.py"), # the cli script
        "synthesize",                     # the subcommand
        "-n 25",                          # only generate 25 samples
        tmp_dir / "titanic.json",         # the input file
        out_file                          # the output file
    ]

    # Run the cli with different extensions
    result = subprocess.run(cmd, check=False)
    assert result.returncode == 0
    assert out_file.is_file()
