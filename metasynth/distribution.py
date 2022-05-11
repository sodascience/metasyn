import numpy as np
from scipy.stats import uniform, norm
from copy import deepcopy
from scipy.stats import randint
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    @classmethod
    def fit(self, series):
        distribution = self._fit(series.dropna())
        return distribution

    @classmethod
    @abstractmethod
    def _fit(self, series):
        pass

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def AIC(self, values):
        return 0.0


class MetaDistribution(ABC):
    @classmethod
    def from_dict(cls, meta_dict):
        meta_dict = deepcopy(meta_dict)
        name = meta_dict.pop("name")
        for dist_type in cls.dist_types:
            if name == dist_type.__name__:
                return dist_type(**meta_dict["parameters"])
        raise ValueError("Cannot find right class.")

    @classmethod
    def fit(cls, values):
        instances = [dist_type.fit(values)
                     for dist_type in cls.dist_types]
        i_min = np.argmin([inst.AIC(values) for inst in instances])
        return instances[i_min]


class BaseNumericDistribution(BaseDistribution):
    def AIC(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpdf(vals+1e-7))

    @property
    def n_par(self):
        return len(self.par)

    def __getattr__(self, attr):
        if attr != "par" and attr in self.par:
            return self.par[attr]
        return super().__getattr__(attr)

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def _fit(cls, values):
        param = cls.dist_class.fit(values[~np.isnan(values)])
        return cls(*param)

    def to_dict(self):
        return {
            "parameters": deepcopy(self.par),
            "name": type(self).__name__
        }

    def draw(self):
        return self.dist.rvs()


class UniformDistribution(BaseNumericDistribution):
    dist_class = uniform

    def __init__(self, min_val, max_val):
        self.par = {"min_val": min_val, "max_val": max_val}
        self.dist = uniform(loc=self.min_val, scale=self.max_val-self.min_val)

    @classmethod
    def _fit(cls, values):
        vals = values[~np.isnan(values)]
        delta = vals.max() - vals.min()
        return cls(vals.min()-1e-3*delta, vals.max()+1e-3*delta)


class NormalDistribution(BaseNumericDistribution):
    dist_class = norm

    def __init__(self, mean, std_dev):
        self.par = {"mean": mean, "std_dev": std_dev}
        self.dist = norm(loc=mean, scale=std_dev)


class DiscreteUniformDistribution(BaseNumericDistribution):
    dist_class = randint

    def __init__(self, low, high):
        self.par = {"low": low, "high": high}
        self.dist = self.dist_class(low=low, high=high+1)

    def AIC(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpmf(vals+1e-7))

    @classmethod
    def _fit(cls, values):
        param = {"low": np.min(values), "high": np.max(values)}
        return cls(**param)


class CatFreqDistribution(BaseDistribution):
    def __init__(self, cat_freq):
        self.cat_freq = cat_freq

    @classmethod
    def _fit(cls, values):
        unq_vals, counts = np.unique(values, return_counts=True)
        return cls(dict(zip(unq_vals, counts)))

    def to_dict(self):
        dist_dict = {}
        dist_dict["name"] = type(self).__name__
        dist_dict["parameters"] = {"cat_freq": self.cat_freq}
        return dist_dict

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        p_vals = np.array([x for x in self.cat_freq.values()])
        p_vals = p_vals/np.sum(p_vals)
        return np.random.choice(np.array(list(self.cat_freq)), p=p_vals)


class StringFreqDistribution(BaseDistribution):
    def __init__(self, str_lengths, p_length, all_char_counts):
        self.str_lengths = str_lengths
        self.p_length = p_length
        self.all_char_counts = all_char_counts

    @classmethod
    def _fit(cls, values):
        values = values.astype(str)
        str_lengths, str_len_counts = np.unique([len(x) for x in values],
                                                return_counts=True)
        p_length = str_len_counts/len(values)
        all_char_counts = []
        for i_chr in range(np.max(str_lengths)):
            cur_chars, cur_char_counts = np.unique(
                [x[i_chr] for x in values if len(x) > i_chr],
                return_counts=True)
            all_char_counts.append((cur_chars,
                                    cur_char_counts/np.sum(cur_char_counts)))
        return cls(str_lengths, p_length, all_char_counts)

    def __str__(self):
        avg_len = np.sum([self.str_lengths[i]*self.p_length[i]
                          for i in range(len(self.str_lengths))])
        return f"String variable: avg # of characters: {avg_len} "

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": {
                "str_lengths": deepcopy(self.str_lengths),
                "p_length": deepcopy(self.p_length),
                "all_char_counts": deepcopy(self.all_char_counts),
            }
        }

    def draw(self):
        cur_str = ""
        str_len = np.random.choice(self.str_lengths, p=self.p_length)
        for i_chr in range(str_len):
            char_choices, p_choices = self.all_char_counts[i_chr]
            cur_str += np.random.choice(char_choices, p=p_choices)
        return cur_str


class FloatDistribution(MetaDistribution):
    dist_types = [UniformDistribution, NormalDistribution]


class IntDistribution(MetaDistribution):
    dist_types = [DiscreteUniformDistribution]


class CategoricalDistribution(MetaDistribution):
    dist_types = [CatFreqDistribution]


class StringDistribution(MetaDistribution):
    dist_types = [StringFreqDistribution]
