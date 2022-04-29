import pandas as pd
from metasynth.distribution import FloatDistribution, CategoricalDistribution
from metasynth.distribution import StringDistribution, IntDistribution


class MetaVar():
    def __init__(self, series):
        self.series = series
        self.distribution = None

    @classmethod
    def detect(cls, series_or_dataframe):
        if isinstance(series_or_dataframe, pd.DataFrame):
            return [MetaVar.detect(series_or_dataframe[col])
                    for col in series_or_dataframe]

        series = series_or_dataframe
        try:
            sub_class = MetaVar.sub_types[pd.api.types.infer_dtype(series)]
        except KeyError:
            raise ValueError(f"Type of column '{series.name}' is not supported")
        return sub_class(series)

    @classmethod
    @property
    def sub_types(self):
        return {
            "categorical": CategoricalVar,
            "string": StringVar,
            "integer": IntVar,
            "floating": FloatVar
        }

    def to_dict(self):
        return {
            "name": self.series.name,
            "type": type(self).__name__,
            "distribution": self.distribution.to_dict(),
        }

    def __str__(self):
        return str({
            "name": self.series.name,
            "type": type(self).__name__,
            "distribution": str(self.distribution),
        })

    def fit(self):
        self.distribution = self.dist_class.fit(self.series)


class IntVar(MetaVar):
    dist_class = IntDistribution


class FloatVar(MetaVar):
    dist_class = FloatDistribution


class StringVar(MetaVar):
    dist_class = StringDistribution


class CategoricalVar(MetaVar):
    dist_class = CategoricalDistribution
