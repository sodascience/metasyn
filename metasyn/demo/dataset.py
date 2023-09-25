"""Load/create different demo datasets."""

from pathlib import Path
import random

try:
    from importlib_resources import files
except ImportError:
    from importlib.resources import files  # type: ignore

import pandas as pd
import numpy as np
import wget

from metasyn.distribution.datetime import UniformDateTimeDistribution, UniformTimeDistribution
from metasyn.distribution.datetime import UniformDateDistribution


def create_titanic_demo(output_fp: Path) -> Path:
    """Create demo dataset for the titanic dataset.

    Arguments
    ---------
    output_fp:
        File to write the demonstration table to.

    Returns
    -------
        Output file location.
    """
    titanic_fp = Path("titanic.csv")
    if output_fp.is_file():
        return output_fp
    if not titanic_fp.is_file():
        wget.download(
            "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
    dframe = pd.read_csv(titanic_fp)
    np.random.seed(1283742)
    random.seed(1928374)

    # Convert Age to a nullable integer.
    dframe["Age"] = dframe["Age"].round().astype("Int64")

    # Add a date column.
    date_dist = UniformDateDistribution.default_distribution()
    dframe["Birthday"] = [date_dist.draw() if np.random.rand() < 0.9 else pd.NA
                          for _ in range(len(dframe))]

    # Add a time column.

    time_dist = UniformTimeDistribution.default_distribution()
    dframe["Board time"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA
                            for _ in range(len(dframe))]

    # Add a datetime column
    time_dist = UniformDateTimeDistribution.default_distribution()
    dframe["Married since"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA
                               for _ in range(len(dframe))]

    dframe["all_NA"] = [pd.NA for _ in range(len(dframe))]
    # Remove some columns for brevity and write to a file.
    dframe = dframe.drop(["SibSp", "Pclass", "Survived"], axis=1)
    dframe.to_csv(output_fp, index=False)
    return output_fp


def demo_file(name: str = "titanic") -> Path:
    """Get the path for a demo data file.

    Arguments
    ---------
    name:
        Name of the demo dataset.

    Returns
    -------
        Path to the dataset.
    """
    file_name = None
    if name == "titanic":
        file_name = "demo_titanic.csv"
    if file_name is None:
        raise ValueError(f"No demonstration dataset with name '{name}'")

    return files(__package__) / file_name
