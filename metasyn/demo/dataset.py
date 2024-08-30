"""Create and retrieve demo datasets."""

# import random
from pathlib import Path

import polars as pl

try:
    from importlib_resources import files
except ImportError:
    from importlib.resources import files  # type: ignore

# import numpy as np
# import wget

# from metasyn.distribution.datetime import (
#     DateTimeUniformDistribution,
#     DateUniformDistribution,
#     TimeUniformDistribution,
# )


# def create_titanic_demo(output_fp: Path) -> Path:
#     """Create demo dataset for the titanic dataset.

#     Arguments
#     ---------
#     output_fp:
#         File to write the demonstration table to.

#     Returns
#     -------
#         Output file location.
#     """
#     titanic_fp = Path("titanic.csv")
#     if Path(output_fp).is_file():
#         return output_fp
#     if not titanic_fp.is_file():
#         wget.download(
#             "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
#     dframe = pd.read_csv(titanic_fp)
#     np.random.seed(1283742)
#     random.seed(1928374)

#     # Convert Age to a nullable integer.
#     dframe["Age"] = dframe["Age"].round().astype("Int64")

#     # Add a date column.
#     date_dist = DateUniformDistribution.default_distribution()
#     dframe["Birthday"] = [date_dist.draw() if np.random.rand() < 0.9 else pd.NA
#                           for _ in range(len(dframe))]

#     # Add a time column.

#     time_dist = TimeUniformDistribution.default_distribution()
#     dframe["Board time"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA
#                             for _ in range(len(dframe))]

#     # Add a datetime column
#     time_dist = DateTimeUniformDistribution.default_distribution()
#     dframe["Married since"] = [time_dist.draw() if np.random.rand() < 0.9 else pd.NA
#                                for _ in range(len(dframe))]

#     dframe["all_NA"] = [pd.NA for _ in range(len(dframe))]
#     # Remove some columns for brevity and write to a file.
#     dframe = dframe.drop(["SibSp", "Pclass", "Survived"], axis=1)
#     dframe.to_csv(output_fp, index=False)
#     return output_fp


def demo_file(name: str = "titanic") -> Path:
    """Get the path for a demo data file.

    There are four options:
        - titanic (Included in pandas, but post-processed to contain more columns)
        - spaceship (CC-BY from https://www.kaggle.com/competitions/spaceship-titanic)
        - fruit (very basic example data from Polars)
        - survey (columns from ESS round 11 Human Values Scale questionnaire for the Netherlands)
    Arguments
    ---------
    name:
        Name of the demo dataset.

    Returns
    -------
        Path to the dataset.

    References
    ----------
    European Social Survey European Research Infrastructure (ESS ERIC). (2024). ESS11 integrated
    file, edition 1.0 [Data set]. Sikt - Norwegian Agency for Shared Services in Education and
    Research. https://doi.org/10.21338/ess11e01_0

    """
    if name == "titanic":
        return files(__package__) / "demo_titanic.csv"
    if name == "spaceship":
        return files(__package__) / "demo_spaceship.csv"
    if name == "fruit":
        return files(__package__) / "demo_fruit.csv"
    if name == "survey":
        return files(__package__) / "demo_survey.csv"

    raise ValueError(
        f"No demonstration dataset with name '{name}'. Options: titanic, spaceship, fruit, survey."
    )


def demo_dataframe(name: str = "titanic") -> pl.DataFrame:
    """Get a demonstration dataset as a prepared polars dataframe.

    There are four options:
        - titanic (Included in pandas, but post-processed to contain more columns)
        - spaceship (CC-BY from https://www.kaggle.com/competitions/spaceship-titanic)
        - fruit (very basic example data from Polars)
        - survey (columns from ESS round 11 Human Values Scale questionnaire for the Netherlands)

    Arguments
    ---------
    name:
        Name of the demo dataset: spaceship, fruit, or titanic.

    Returns
    -------
        Polars dataframe with correct column types

    References
    ----------
    European Social Survey European Research Infrastructure (ESS ERIC). (2024). ESS11 integrated
    file, edition 1.0 [Data set]. Sikt - Norwegian Agency for Shared Services in Education and
    Research. https://doi.org/10.21338/ess11e01_0

    """
    file_path = demo_file(name=name)
    if name == "spaceship":
        # the Kaggle spaceship data (CC-BY)
        data_types = {
            "HomePlanet": pl.Categorical,
            "CryoSleep": pl.Categorical,
            "VIP": pl.Categorical,
            "Destination": pl.Categorical,
            "Transported": pl.Categorical,
        }
        return pl.read_csv(file_path, schema_overrides=data_types, try_parse_dates=True)
    if name == "titanic":
        # our edited titanic data
        data_types = {"Sex": pl.Categorical, "Embarked": pl.Categorical}
        return pl.read_csv(file_path, schema_overrides=data_types, try_parse_dates=True)
    if name == "fruit":
        # basic fruit data from polars example
        data_types = {"fruits": pl.Categorical, "cars": pl.Categorical}
        return pl.read_csv(file_path, schema_overrides=data_types)
    if name == "survey":
        return pl.read_csv(file_path)

    raise ValueError(
        f"No demonstration dataset with name '{name}'. Options: titanic, spaceship, fruit."
    )
