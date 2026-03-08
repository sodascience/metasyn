"""Create and retrieve demo datasets."""

# import random
import string
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta
from importlib.resources import files
from pathlib import Path

import faker
import numpy as np
import polars as pl

from metasyn.varspec import VarSpec

_AVAILABLE_DATASETS = {}


def register(*args):
    """Register a dataset so that it can be found by name."""

    def _wrap(cls):
        _AVAILABLE_DATASETS[cls().name] = cls()
        return cls

    return _wrap(*args)


class BaseDataset(ABC):
    """Base class for demo datasets."""

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def file_location(self):
        return files(__package__) / f"demo_{self.name}.csv"

    def get_dataframe(self):
        return pl.read_csv(self.file_location, schema_overrides=self.schema, try_parse_dates=True)

    @property
    @abstractmethod
    def schema(self):
        pass

    @property
    def var_specs(self):
        return []


class BaseMultiDataset(BaseDataset):
    """Abstract class to define a dataset with multiple tables."""

    def get_dataframe(self):
        """Alias for get_dataframes()."""
        return self.get_dataframes()

    @property
    @abstractmethod
    def file_location(self):
        pass

    def get_dataframes(self):
        """Create the dataframes (from file for example).

        Returns
        -------
        dataframes:
            Dictionary with dataframes.
        """
        return {name: pl.read_csv(path, schema_overrides=self.schema, try_parse_dates=True)
                for name, path in self.file_location.items()}


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
class SyntheaImagingDataset(BaseDataset):
    """Synthetic medical health dataset from Synthea.

    Jason Walonoski, Mark Kramer, Joseph Nichols, Andre Quina, Chris Moesel, Dylan Hall,
    Carlton Duffett, Kudakwashe Dube, Thomas Gallagher, Scott McLachlan,
    Synthea: An approach, method, and software mechanism for generating synthetic patients
    and the synthetic electronic health care record, Journal of the American Medical Informatics
    Association, Volume 25, Issue 3, March 2018, Pages 230–238, https://doi.org/10.1093/jamia/ocx079
    """

    @property
    def name(self):
        return "synthea_imaging"

    @property
    def schema(self):
        return {"SOP_DESCRIPTION": pl.Categorical, "BODYSITE_DESCRIPTION": pl.Categorical}


@register
class HospitalDataset(BaseDataset):
    """Example electronic health record hospital dataset.

    This dataset was created manually by the metasyn team.
    """

    @property
    def name(self):
        return "hospital"

    @property
    def schema(self):
        return {"date_admitted": pl.Date, "time_admitted": pl.Time, "type": pl.Categorical}


@register
class DrugUseDataset(BaseDataset):
    """Example dataset with answers to an open question on study participants' daily drug use.

    This example dataset was generated through ChatGPT-4o on 07-11-2024 using the following prompt:
    > Create a csv with 12 rows and 2 columns: participant_id, and drug_use. The participant_id
    has a standard alphanumeric structure, and the drug_use contains participant's responses on
    how they use drugs in their daily life.
    """

    @property
    def name(self):
        return "druguse"

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
        return {
            col_name: (getattr(pl, col_name) if col_name != "NA" else pl.String)
            for col_name in columns
        }

    @classmethod
    def create(cls, csv_file):
        all_series = []
        n_rows = 100

        for int_val in [8, 16, 32, 64]:
            all_series.append(
                pl.Series(
                    f"Int{int_val}",
                    [np.random.randint(-10, 10) for _ in range(n_rows)],
                    dtype=getattr(pl, f"Int{int_val}"),
                )
            )
            all_series.append(
                pl.Series(
                    f"UInt{int_val}",
                    [np.random.randint(10) for _ in range(n_rows)],
                    dtype=getattr(pl, f"UInt{int_val}"),
                )
            )

        for float_val in [32, 64]:
            all_series.append(
                pl.Series(
                    f"Float{float_val}",
                    np.random.randn(n_rows),
                    dtype=getattr(pl, f"Float{float_val}"),
                )
            )

        all_series.append(
            pl.Series(
                "Date",
                [date(2024, 9, 4) + timedelta(days=i) for i in range(n_rows)],
                dtype=pl.Date,
            )
        )
        all_series.append(
            pl.Series(
                "Datetime",
                [
                    datetime(2024, 9, 4, 12, 30, 12)
                    + timedelta(hours=i, minutes=i * 2, seconds=i * 3)
                    for i in range(n_rows)
                ],
                dtype=pl.Datetime,
            )
        )
        all_series.append(
            pl.Series(
                "Time",
                [time(3 + i // 20, 6 + i // 12, 12 + i // 35) for i in range(n_rows)],
                dtype=pl.Time,
            )
        )
        all_series.append(
            pl.Series(
                "String", np.random.choice(list(string.printable), size=n_rows), dtype=pl.String
            )
        )
        all_series.append(
            pl.Series(
                "Utf8", np.random.choice(list(string.printable), size=n_rows), dtype=pl.Utf8
            )
        )
        all_series.append(
            pl.Series(
                "Categorical",
                np.random.choice(list(string.ascii_uppercase[:5]), size=n_rows),
                dtype=pl.Categorical,
            )
        )
        all_series.append(
            pl.Series("Boolean", np.random.choice([True, False], size=n_rows), dtype=pl.Boolean)
        )
        all_series.append(pl.Series("NA", [None for _ in range(n_rows)], dtype=pl.String))

        # Add NA's for all series except the categorical
        for series in all_series:
            if series.name != "Categorical":
                none_idx = np.random.choice(np.arange(n_rows), size=n_rows // 10, replace=False)
                none_idx.sort()
                series[none_idx] = None

        # Write to a csv file
        pl.DataFrame(all_series).write_csv(csv_file)

@register
class ShopMultiDataset(BaseMultiDataset):
    """An example dataset containing customers, products and purchases."""

    @property
    def name(self):
        return "shop_multi"

    @property
    def schema(self):
        return {}

    @property
    def file_location(self):
        return {
            "customers": files(__package__) / self.name / "customers.csv",
            "products": files(__package__) / self.name / "products.csv",
            "purchases": files(__package__) / self.name / "purchases.csv",
        }

    @classmethod
    def create(cls, data_dir: Path, n_user: int = 200, n_product: int = 500,
               n_purchases: int = 1000):
        user_ids = np.unique(np.random.randint(123, 123456+100*n_user, size=2*n_user))[:n_user]
        np.random.shuffle(user_ids)

        product_ids = np.unique(np.random.randint(123, 123456+100*n_product, size=2*n_product)
                                )[:n_product]
        np.random.shuffle(product_ids)

        fake = faker.Faker()
        df_customers = pl.DataFrame(
            {
                "id": user_ids,
                "address": [fake.address() for _ in range(n_user)],
                "credit_card_nr": [fake.credit_card_number() for _ in range(n_user)],
                "signup_date": [fake.date() for _ in range (n_user)]
            }
        )
        df_customers

        df_products = pl.DataFrame(
            {
                "id": product_ids,
                "name": [fake.word() for _ in range(n_product)],
                "current_price":  [fake.pricetag() for _ in range(n_product)],
                "stock": np.random.randint(0, 10, size=n_product)
            }
        )

        df_purchases = pl.DataFrame(
            {
                "id": np.arange(n_purchases),
                "customer_id": df_customers["id"].sample(n_purchases, with_replacement=True,
                                                         shuffle=True),
                "price_paid": [fake.pricetag() for _ in range(n_purchases)],
                "product_id": df_products["id"].sample(n_purchases, with_replacement=True,
                                                       shuffle=True),
            }
        )
        df_products.write_csv(data_dir / "products.csv")
        df_purchases.write_csv(data_dir / "purchases.csv")
        df_customers.write_csv(data_dir / "customers.csv")


def _get_demo_class(name):
    if name in _AVAILABLE_DATASETS:
        return _AVAILABLE_DATASETS[name]
    raise ValueError(
        f"No demonstration dataset with name '{name}'. Options: {list(_AVAILABLE_DATASETS)}."
    )


def demo_file(name: str = "titanic") -> Path:
    """Get the path for a demo data file.

    There are six options:
        - titanic (Included in pandas, but post-processed to contain more columns)
        - spaceship (CC-BY from https://www.kaggle.com/competitions/spaceship-titanic)
        - synthea_imaging (CC-BY from https://synthea.mitre.org/downloads)
        - fruit (very basic example data from Polars)
        - survey (columns from ESS round 11 Human Values Scale questionnaire for the Netherlands)
        - test (columns with all supported data types)

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

    There are six options:
        - titanic (Included in pandas, but post-processed to contain more columns)
        - spaceship (CC-BY from https://www.kaggle.com/competitions/spaceship-titanic)
        - synthea_imaging (CC-BY from https://synthea.mitre.org/downloads)
        - fruit (very basic example data from Polars)
        - survey (columns from ESS round 11 Human Values Scale questionnaire for the Netherlands)
        - test (columns with all supported data types)

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
