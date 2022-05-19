"""Variable module that creates metadata variables."""  # pylint: disable=invalid-name

import pandas as pd
from metasynth.metadist import FloatDistribution, CategoricalDistribution
from metasynth.metadist import StringDistribution, IntDistribution


class MetaVar():
    """Meta data variable.

    Acts as a base class for specific types of variables, but also as a
    launching pad for detecting its type.

    Parameters
    ----------
    series : pd.Series
        Series to create the variable from. Series is None by default and in
        this case the value is ignored. If it is not supplied, then the
        variable cannot be fit.
    name : str
        Name of the variable/column.
    distribution : BaseDistribution
        Distribution to draw random values from. Can also be set by using the
        fit method.
    prop_missing : float
        Proportion of the series that are missing/NA.
    """

    dist_class = None
    dtype = "unknown"

    def __init__(self, series=None, name=None, distribution=None, prop_missing=0, dtype=None):
        if series is None:
            self.name = name
            self.prop_missing = prop_missing
            if dtype is not None:
                self.dtype = dtype
        else:
            self.name = series.name
            self.prop_missing = (len(series) - len(series.dropna()))/len(series)
            self.dtype = series.dtype

        self.series = series
        self.distribution = distribution

    @classmethod
    def detect(cls, series_or_dataframe):
        """Detect variable class(es) of series or dataframe.

        Parameters
        ----------
        series_or_dataframe: Union[pd.Series, pd.Dataframe]
            If the variable is a pandas Series, then find the correct
            variable type and create an instance of that variable.
            If a Dataframe is supplied instead, a list of of variables is
            returned: one for each column in the dataframe.

        Returns
        -------
        MetaVar:
            It returns a meta data variable of the correct type.
        """
        if isinstance(series_or_dataframe, pd.DataFrame):
            return [MetaVar.detect(series_or_dataframe[col])
                    for col in series_or_dataframe]

        series = series_or_dataframe
        try:
            sub_class = MetaVar.sub_types[pd.api.types.infer_dtype(series)]  # pylint: disable=unsubscriptable-object
        except KeyError as e:
            raise ValueError(f"Type of column '{series.name}' is not supported") from e
        return sub_class(series)

    @classmethod
    @property
    def sub_types(cls):
        """Return a dictionary to translate type names to VarTypes."""
        return {
            "categorical": CategoricalVar,
            "string": StringVar,
            "integer": IntVar,
            "floating": FloatVar
        }

    def to_dict(self):
        """Create a dictionary from the variable."""
        return {
            "name": self.name,
            "type": type(self).__name__,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": self.distribution.to_dict(),
        }

    def __str__(self):
        """Create a readable string from a variable."""
        return str({
            "name": self.name,
            "type": type(self).__name__,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": str(self.distribution),
        })

    def fit(self):
        """Fit distributions to the data.

        If multiple distributions are available for the current data type,
        use the one that fits the data the best.

        While it has no arguments or return values, it will set the
        distribution attribute to the most suitable distribution.
        """
        if self.series is None:
            raise ValueError("Cannot fit distribution if we don't have the"
                             "original data.")
        self.distribution = self.dist_class.fit(self.series)

    def draw(self):
        """Draw a random item for the variable in whatever type is required."""
        if self.distribution is None:
            raise ValueError("Cannot draw without distribution")
        return self.distribution.draw()

    def draw_series(self, n):
        """Draw a new synthetic series from the metadata.

        Parameters
        ----------
        n: int
            Length of the series to be created.

        Returns
        -------
        pandas.Series:
            Pandas series with the synthetic data.
        """
        return pd.Series([self.draw() for _ in range(n)], dtype=self.dtype)

    @classmethod
    def from_dict(cls, var_dict):
        """Restore variable from dictionary.

        Parameters
        ----------
        var_dict: dict
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        for meta_class in cls.sub_types.values():
            if meta_class.__name__ == var_dict["type"]:
                return meta_class(
                    name=var_dict["name"],
                    distribution=meta_class.dist_class.from_dict(var_dict["distribution"]),
                    prop_missing=var_dict["prop_missing"], dtype = var_dict["dtype"])
        raise ValueError("Cannot find meta class '{var_dict['type']}'.")


class IntVar(MetaVar):
    """Integer variable class."""

    dist_class = IntDistribution
    dtype = "int"

class FloatVar(MetaVar):
    """Floating point variable class."""

    dist_class = FloatDistribution
    dtype = "float"

class StringVar(MetaVar):
    """String variable class."""

    dist_class = StringDistribution
    dtype = "str"

class CategoricalVar(MetaVar):
    """Categorical variable class."""

    dist_class = CategoricalDistribution
    dtype = "category"
