from random import random
from pathlib import Path

import pytest
import pandas as pd

from metasynth.dataset import MetaDataset
from metasynth.var import MetaVar
from metasynth.disttree import get_disttree


def test_dataset(tmp_path):
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
    titanic_fp = Path("tests", "data", "titanic.csv")
    tmp_fp = tmp_path / "tmp.json"
    df = pd.read_csv(titanic_fp, dtype=dtypes)
    dataset = MetaDataset.from_dataframe(
        df.iloc[:100],
        spec={
                "Name": {"prop_missing": 0.5},
                "Ticket": {"description": "test_description"},
                "Fare": {"distribution": "normal"},
                "PassengerId": {"unique": True},
             })

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

    check_dataset(dataset)
    dataset.to_json(tmp_fp)
    dataset = MetaDataset.from_json(tmp_fp)
    check_dataset(dataset)

    dataset.descriptions = {"Embarked": "Some description", "Sex": "Other description"}
    assert dataset.descriptions["Embarked"] == "Some description"
    with pytest.raises(AssertionError):
        dataset.descriptions = "12345"
    dataset.descriptions = {"Embarked": "New description"}
    assert dataset.descriptions["Embarked"] == "New description"
    assert dataset.descriptions["Sex"] == "Other description"
    dataset.descriptions = list(df)
    for name in list(df):
        assert dataset.descriptions[name] == name


def test_distributions(tmp_path):
    tmp_fp = tmp_path / "tmp.json"

    dist_tree = get_disttree()
    for var_type in dist_tree.all_var_types:
        for dist in dist_tree.get_dist_list(var_type):
            var = MetaVar(var_type, name="None", distribution=dist.default_distribution(),
                          prop_missing=random())
            dataset = MetaDataset([var], n_rows=10)
            dataset.to_json(tmp_fp)
