import subprocess
from pathlib import Path
from pytest import mark, fixture
import polars as pl
from metasynth import MetaFrame

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
    input_file = tmp_dir / "titanic.json"
    output_file = tmp_dir / f"titanic{ext}"

    # Run the cli with different extensions
    result = subprocess.run(f"metasynth -n 25 {input_file} {output_file}", check=False)
    assert result.returncode == 0
    assert output_file.is_file()
