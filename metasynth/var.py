import pandas as pd
from metasynth.distribution import FloatDistribution, CategoricalDistribution
from metasynth.distribution import StringDistribution, IntDistribution


class MetaVar():
    """"""
    def __init__(self, series=None, name=None, distribution=None, prop_missing=0):
        if series is None:
            self.name = name
            self.prop_missing = prop_missing
        else:
            self.name = series.name
            self.series = series
            self.prop_missing = (len(series) - len(series.dropna()))/len(series)

        self.distribution = distribution

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
            "name": self.name,
            "type": type(self).__name__,
            "prop_missing": self.prop_missing,
            "distribution": self.distribution.to_dict(),
        }

    def __str__(self):
        return str({
            "name": self.name,
            "type": type(self).__name__,
            "prop_missing": self.prop_missing,
            "distribution": str(self.distribution),
        })

    def fit(self):
        self.distribution = self.dist_class.fit(self.series)

    def draw(self):
        if self.distribution is None:
            raise ValueError("Cannot draw without distribution")
        return self.distribution.draw()

    @classmethod
    def from_dict(cls, var_dict):
        for meta_class in cls.sub_types.values():
            if meta_class.__name__ == var_dict["type"]:
                return meta_class(
                    name=var_dict["name"],
                    distribution=meta_class.dist_class.from_dict(var_dict["distribution"]),
                    prop_missing=var_dict["prop_missing"])
        raise ValueError("Cannot find meta class '{var_dict['type']}'.")


class IntVar(MetaVar):
    dist_class = IntDistribution


class FloatVar(MetaVar):
    dist_class = FloatDistribution


class StringVar(MetaVar):
    dist_class = StringDistribution


class CategoricalVar(MetaVar):
    dist_class = CategoricalDistribution
