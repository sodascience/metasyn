import numpy as np


class MetaVar():
    @classmethod
    def detect(cls, series, series_str):
        max_detect = -1
        detect_var = None
        for detect_cls in cls().sub_types:
            new_detect, new_cls = detect_cls.detect(series, series_str)
            if new_detect > max_detect:
                max_detect = new_detect
                detect_var = new_cls
        return max_detect, detect_var

    @property
    def sub_types(self):
        return [NumericalVar, NanVar, StringVar, CategoricalVar]


class NumericalVar(MetaVar):
    @property
    def sub_types(self):
        return [IntVar, FloatVar]


class IntVar(NumericalVar):
    var_strength = 2

    @classmethod
    def detect(cls, series, series_str):
        if str(series.dtype).startswith("int"):
            return cls.var_strength, cls
        if not str(series.dtype).startswith("float"):
            return 0, cls
        values = series.values[~np.isnan(series.values)]
        if np.all(np.fabs(values - values.astype(int)) < 1e-7):
            return cls.var_strength, cls
        return 0, cls


class FloatVar(NumericalVar):
    var_strength = 1

    @classmethod
    def detect(cls, series, series_str):
        if str(series.dtype).startswith("float"):
            if np.sum(np.isnan(series.values)) < len(series):
                return cls.var_strength, cls
            else:
                return cls.var_strength/2, cls
        return 0, cls


class NanVar(MetaVar):
    var_strength = 3

    @classmethod
    def detect(cls, series, series_str):
        if str(series.dtype).startswith("float"):
            if np.sum(np.isnan(series.values)) == len(series):
                return cls.var_strength, cls
        return 0, cls


class StringVar(MetaVar):
    var_strength = 0.5

    @classmethod
    def detect(cls, series, series_str):
        return 0.5, cls


class CategoricalVar(MetaVar):
    var_strength = 3

    @classmethod
    def detect(cls, series, series_str):
        series = series.dropna()
        series_str = series.dropna()
        if len(series) == 0:
            return 0, cls
        int_score, _ = IntVar.detect(series, series_str)
        float_score, _ = FloatVar.detect(series, series_str)
        if int_score < float_score:
            return 0, cls

        if int_score > 0:
            values = series.values.astype(int)
        else:
            values = series.values.astype(str)
        n_unique = len(np.unique(values))
        if len(series) > 20 and n_unique**2 < len(series):
            return cls.var_strength, cls
        return 0, cls
    
    