"""Create and retrieve demo datasets."""

# import random
import string
from abc import ABC, abstractproperty
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from metasyn.varspec import VarSpec

try:
    from importlib_resources import files
except ImportError:
    from importlib.resources import files  # type: ignore

_AVAILABLE_DATASETS = {}


def register(*args):
    """Register a dataset so that it can be found by name."""
    def _wrap(cls):
        _AVAILABLE_DATASETS[cls().name] = cls()
        return cls
    return _wrap(*args)


class BaseDataset(ABC):
    """Base class for demo datasets."""

    @abstractproperty
    def name(self):
        pass

    @property
    def file_location(self):
        return files(__package__) / f"demo_{self.name}.csv"

    def get_dataframe(self):
        return pl.read_csv(self.file_location, schema_overrides=self.schema,
                           try_parse_dates=True)

    @abstractproperty
    def schema(self):
        pass

    @property
    def var_specs(self):
        return []


@register
class TitanicDataset(BaseDataset):
    """Included in pandas, but post-processed to contain more columns."""

    @property
    def name(self):
        return "titanic"

    @property
    def schema(self):
        return {"Sex": pl.Categorical, "Embarked": pl.Categorical}

    @property
    def var_specs(self):
        return [VarSpec("PassengerId", unique=True)]

@register
class SpaceShipDataset(BaseDataset):
    """CC-BY from https://www.kaggle.com/competitions/spaceship-titanic."""

    @property
    def name(self):
        return "spaceship"

    @property
    def schema(self):
        return {
            "HomePlanet": pl.Categorical,
            "CryoSleep": pl.Categorical,
            "VIP": pl.Categorical,
            "Destination": pl.Categorical,
            "Transported": pl.Categorical,
        }


@register
class FruitDataset(BaseDataset):
    """Very basic example data from Polars."""

    @property
    def name(self):
        return "fruit"

    @property
    def schema(self):
        return {"fruits": pl.Categorical, "cars": pl.Categorical}

    @property
    def var_specs(self):
        return [VarSpec("ID", unique=True), VarSpec("B", unique=False)]


@register
class SurveyDataset(BaseDataset):
    """Columns from ESS round 11 Human Values Scale questionnaire for the Netherlands."""

    @property
    def name(self):
        return "survey"

    @property
    def schema(self):
        return {}


@register
class TestDataset(BaseDataset):
    """Test dataset with all supported data types."""

    @property
    def name(self):
        return "test"

    @property
    def schema(self):
        columns = pl.read_csv(self.file_location).columns
        return {col_name: (getattr(pl, col_name[3:]) if col_name != "NA" else pl.String)
                for col_name in columns}

    @classmethod
    def create(cls, csv_file):
        all_series = []
        n_rows = 100

        for int_val in [8, 16, 32, 64]:
            all_series.append(pl.Series(f"pl.Int{int_val}",
                                        [np.random.randint(-10, 10) for _ in range(n_rows)],
                                        dtype=getattr(pl, f"Int{int_val}")))
            all_series.append(pl.Series(f"pl.UInt{int_val}",
                                        [np.random.randint(10) for _ in range(n_rows)],
                                        dtype=getattr(pl, f"UInt{int_val}")))

        for float_val in [32, 64]:
            all_series.append(pl.Series(f"pl.Float{float_val}",
                                        np.random.randn(n_rows),
                                        dtype=getattr(pl, f"Float{float_val}")))

        all_series.append(pl.Series("pl.Date", [date(2024, 9, 4) + timedelta(days=i)
                                                for i in range(n_rows)],
                                    dtype=pl.Date))
        all_series.append(pl.Series("pl.Datetime",
                                    [datetime(2024, 9, 4, 12, 30, 12)
                                        + timedelta(hours=i, minutes=i*2, seconds=i*3)
                                        for i in range(n_rows)],
                                    dtype=pl.Datetime))
        all_series.append(pl.Series("pl.Time",
                                    [time(3+i//20, 6+i//12, 12+i//35) for i in range(n_rows)],
                                    dtype=pl.Time))
        all_series.append(pl.Series("pl.String",
                                    np.random.choice(list(string.printable), size=n_rows),
                                    dtype=pl.String))
        all_series.append(pl.Series("pl.Utf8",
                                    np.random.choice(list(string.printable), size=n_rows),
                                    dtype=pl.Utf8))
        all_series.append(pl.Series("pl.Categorical",
                                    np.random.choice(list(string.ascii_uppercase[:5]), size=n_rows),
                                    dtype=pl.Categorical))
        all_series.append(pl.Series("NA", [None for _ in range(n_rows)], dtype=pl.String))

        # Add NA's for all series except the categorical
        for series in all_series:
            if series.name != "pl.Categorical":
                none_idx = np.random.choice(np.arange(n_rows), size=n_rows//10, replace=False)
                none_idx.sort()
                series[none_idx] = None

        # Write to a csv file
        pl.DataFrame(all_series).write_csv(csv_file)


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
