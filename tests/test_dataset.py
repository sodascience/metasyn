"""Testing module for fitting dataframes for both pandas and polars."""
from pathlib import Path
from random import random

import pandas as pd
import polars as pl
import pytest
from pytest import mark

from metasyn.demo.dataset import _AVAILABLE_DATASETS, _get_demo_class, demo_dataframe, demo_file
from metasyn.metaframe import MetaFrame
from metasyn.privacy import BasicPrivacy
from metasyn.registry import DistributionRegistry
from metasyn.var import MetaVar

dtypes = {
    "PassengerId": "int",
    "Survived": "category",
    "Pclass": "category",
    "Name": "string",
    "Sex": "category",
    "SibSp": "category",
    "Parch": "category",
    "Ticket": "string",
    "Cabin": "string",
    "Embarked": "category",
    "Age": "float",
    "Fare": "float",
}


def _read_csv(fp, dataframe_lib):
    if dataframe_lib == "pandas":
        df = pd.read_csv(fp, dtype=dtypes)
        return df.iloc[:100]
    else:
        df = pl.read_csv(fp, schema_overrides={x: pl.Categorical for x, x_type in dtypes.items()
                                               if x_type == "category"})
        return df[:100]


@mark.parametrize("dataframe_lib", ["polars", "pandas"])
def test_dataset(tmp_path, dataframe_lib):
    """Integration test that loads the CSV, and creates a MetaFrame from that."""
    titanic_fp = Path("tests", "data", "titanic.csv")
    tmp_fp = tmp_path / "tmp.json"
    df = _read_csv(titanic_fp, dataframe_lib)
    dataset = MetaFrame.fit_dataframe(
        df,
        var_specs=[
                {"name": "Name", "prop_missing": 0.5},
                {"name": "Ticket", "description": "test_description"},
                {"name": "Fare", "distribution": {"name": "normal"}},
                {"name": "PassengerId", "distribution": {"unique": True}},
             ])

    def check_dataset(dataset):
        assert dataset.n_columns == 12
        assert dataset.n_rows == 100
        sub_types = {
            "int": "discrete",
            "string": "string",
            "category": "categorical",
            "float": "continuous",
        }
        for col, type_name in dtypes.items():
            meta_var_type = sub_types[type_name]
            assert dataset[col].var_type == meta_var_type
        with pytest.raises(KeyError):
            dataset["unknown"]
        with pytest.raises(IndexError):
            dataset[20]
        with pytest.raises(TypeError):
            dataset[MetaVar]

        assert "Rows" in str(dataset)
        assert len(dataset.synthesize()) == dataset.n_rows
        assert len(dataset.synthesize(5)) == 5
        with pytest.raises(ValueError):
            dataset.n_rows = None
            dataset.synthesize()
        dataset.n_rows = 100

    check_dataset(dataset)
    dataset.save_json(tmp_fp)
    dataset = MetaFrame.load_json(tmp_fp)
    check_dataset(dataset)

    # Check the description functionality of metasyn.
    dataset.descriptions = {"Embarked": "Some description", "Sex": "Other description"}
    assert dataset.descriptions["Embarked"] == "Some description"
    with pytest.raises(AssertionError):
        dataset.descriptions = "12345"
    dataset.descriptions = {"Embarked": "New description"}
    assert dataset.descriptions["Embarked"] == "New description"
    assert dataset.descriptions["Sex"] == "Other description"
    if isinstance(df, pl.DataFrame):
        col_names = df.columns
    else:
        col_names = list(df)
    dataset.descriptions = col_names
    for name in col_names:
        print(name, dataset.descriptions[name])
        assert dataset.descriptions[name] == name

    assert isinstance(repr(dataset), str)

    # Check whether non-columns raise an error
    with pytest.raises(ValueError):
        dataset = MetaFrame.fit_dataframe(df, var_specs=[{"name": "unicorn", "prop_missing": 0.5}])


def test_distributions(tmp_path):
    """Create all available distributions and save a metaframe with it."""
    tmp_fp = tmp_path / "tmp.json"

    registry = DistributionRegistry.parse("builtin")
    var_type_set = set()
    for fitter in registry.fitters:
        if isinstance(fitter.var_type, str):
            var_type_set.add(fitter.var_type)
        else:
            var_type_set.update(fitter.var_type)
    for var_type in var_type_set:
        for fitter in registry.filter_fitters(BasicPrivacy(), var_type):
            var = MetaVar(name="None", var_type=var_type,
                          distribution=fitter.distribution.default_distribution(),
                          prop_missing=random())
            dataset = MetaFrame([var], n_rows=10)
            dataset.save_json(tmp_fp)

@mark.parametrize(
    "dataset_name", list(_AVAILABLE_DATASETS)
)
def test_demo_datasets(tmp_path, dataset_name):
    """Test all built-in demo datasets and see if they can be synthesized."""
    demo_fp = demo_file(dataset_name)
    demo_df = demo_dataframe(dataset_name)
    demo_class = _get_demo_class(dataset_name)

    assert demo_fp.is_file()
    assert isinstance(demo_df, pl.DataFrame)

    mf = MetaFrame.fit_dataframe(demo_df, var_specs=demo_class.var_specs)
    assert isinstance(mf, MetaFrame)

    tmp_file = tmp_path / "gmf.json"
    mf.save_json(tmp_file)
    mf = MetaFrame.load_json(tmp_file)

    df_syn = mf.synthesize(100)
    df_syn_1 = mf.synthesize(100, seed=1234)
    df_syn_2 = mf.synthesize(100, seed=1234)
    for col in df_syn.columns:
        assert all(df_syn_1[col].drop_nulls() == df_syn_2[col].drop_nulls())

    for col, dtype in demo_class.schema.items():
        assert dtype == df_syn[col].dtype


def test_demo_non_exist():
    """Check that trying to get a demo dataset that doesn't exist raises an error."""
    with pytest.raises(ValueError):
        demo_file("non-existing-dataset")


def test_create_dataset(tmpdir):
    """Test re-creation of the test dataset."""
    test_class = _get_demo_class("test")
    test_class.create(tmpdir/"test.csv")
    df = pl.read_csv(Path(tmpdir/"test.csv"))
    assert isinstance(df, pl.DataFrame)
