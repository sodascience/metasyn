"""Create and retrieve demo datasets."""

# import random
from abc import ABC, abstractclassmethod
from pathlib import Path

import polars as pl

try:
    from importlib_resources import files
except ImportError:
    from importlib.resources import files  # type: ignore

_AVAILABLE_DATASETS = {}


def register(*args):
    def _wrap(cls):
        _AVAILABLE_DATASETS[cls.name] = cls
        return cls
    return _wrap(*args)


class BaseDataset(ABC):

    @property
    @abstractclassmethod
    def name(cls):
        pass

    @classmethod
    @property
    def file_location(cls):
        return files(__package__) / f"demo_{cls.name}.csv"

    @abstractclassmethod
    def get_dataframe(cls):
        return pl.read_csv(cls.file_location)


@register
class TitanicDataset(BaseDataset):
    @classmethod
    @property
    def name(cls):
        return "titanic"

    @classmethod
    def get_dataframe(cls):
        file_path = cls.file_location
        data_types = {"Sex": pl.Categorical, "Embarked": pl.Categorical}
        return pl.read_csv(file_path, schema_overrides=data_types, try_parse_dates=True)


@register
class SpaceShipDataset(BaseDataset):
    @classmethod
    @property
    def name(cls):
        return "spaceship"

    @classmethod
    def get_dataframe(cls):
        # the Kaggle spaceship data (CC-BY)
        data_types = {
            "HomePlanet": pl.Categorical,
            "CryoSleep": pl.Categorical,
            "VIP": pl.Categorical,
            "Destination": pl.Categorical,
            "Transported": pl.Categorical,
        }
        return pl.read_csv(
            cls.file_location, schema_overrides=data_types, try_parse_dates=True
        )


@register
class FruitDataset(BaseDataset):
    @classmethod
    @property
    def name(cls):
        return "fruit"

    @classmethod
    def get_dataframe(cls):
        # basic fruit data from polars example
        data_types = {"fruits": pl.Categorical, "cars": pl.Categorical}
        return pl.read_csv(cls.file_location, schema_overrides=data_types)


@register
class SurveyDataset(BaseDataset):
    @classmethod
    @property
    def name(cls):
        return "survey"

    @classmethod
    def get_dataframe(cls):
        return super().get_dataframe()


@register
class TestDataset(BaseDataset):
    @classmethod
    @property
    def name(cls):
        return "test"

    @classmethod
    def create(cls, file_out):
        pass

def _get_demo_class(name):
    if name in _AVAILABLE_DATASETS:
        return _AVAILABLE_DATASETS[name]
    raise ValueError(
        f"No demonstration dataset with name '{name}'. Options: {list(_AVAILABLE_DATASETS)}."
    )


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
    return _get_demo_class(name).file_location


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
    return _get_demo_class(name).get_dataframe()
