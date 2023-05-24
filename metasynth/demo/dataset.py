"""Load/create different demo datasets."""

from pathlib import Path
import random

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import pandas as pd
import polars as pl
import numpy as np
import wget

import metasynth
from metasynth.distribution.datetime import UniformDateTimeDistribution, UniformTimeDistribution
from metasynth.distribution.datetime import UniformDateDistribution


def create_titanic_demo():
    demonstration_fp = Path("demonstration.csv")
    titanic_fp = Path("titanic.csv")
    if demonstration_fp.is_file():
        return demonstration_fp
    if not titanic_fp.is_file():
        wget.download("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
    df = pd.read_csv(titanic_fp)
    np.random.seed(1283742)
    random.seed(1928374)

    # Convert Age to a nullable integer.
    df["Age"] = df["Age"].round().astype("Int64")

    # Add a date column.
    date_dist = UniformDateDistribution.default_distribution()
    df["Birthday"] = [date_dist.draw() if np.random.rand() < 0.9 else pd.NA for _ in range(len(df))]

    # Add a time column.

    time_dist = UniformTimeDistribution.default_distribution()
    df["Board time"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA for _ in range(len(df))]

    # Add a datetime column
    time_dist = UniformDateTimeDistribution.default_distribution()
    df["Married since"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA for _ in range(len(df))]

    df["all_NA"] = [pd.NA for _ in range(len(df))]
    # Remove some columns for brevity and write to a file.
    df = df.drop(["SibSp", "Pclass", "Ticket", "Survived"], axis=1)
    df.to_csv(demonstration_fp, index=False)
    return pl.DataFrame(df)


def demo_file(name="titanic"):
    file_fp = None
    if name == "titanic":
        file_name = "demo_titanic.csv"
    if file_fp is None:
        raise ValueError(f"No demonstration dataset with name '{name}'")

    return files(metasynth.demo / file_name)
