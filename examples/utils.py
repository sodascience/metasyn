import pandas as pd
from metasynth.distribution.datetime import UniformDateTimeDistribution, UniformTimeDistribution
from metasynth.distribution.datetime import UniformDateDistribution
import wget
from pathlib import Path


def get_demonstration_fp():
    demonstration_fp = Path("demonstration.csv")
    if demonstration_fp.is_file():
        return demonstration_fp
    wget.download("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
    df = pd.read_csv("titanic.csv")

    # Convert Age to a nullable integer.
    df["Age"] = df["Age"].round().astype("Int64")

    # Add a date column.
    date_dist = UniformDateDistribution._example_distribution()
    df["Birthday"] = [date_dist.draw() for _ in range(len(df))]

    # Add a time column.

    time_dist = UniformTimeDistribution._example_distribution()
    df["Board time"] = [time_dist.draw() for _ in range(len(df))]

    # Add a datetime column
    time_dist = UniformDateTimeDistribution._example_distribution()
    df["Married since"] = [time_dist.draw() for _ in range(len(df))]

    # Remove some columns for brevity and write to a file.
    df = df.drop(["SibSp", "Pclass", "Ticket", "Survived"], axis=1)
    df.to_csv(demonstration_fp, index=False)
    return demonstration_fp
