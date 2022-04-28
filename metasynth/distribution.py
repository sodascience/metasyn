import numpy as np
from scipy.stats import uniform, norm
from copy import deepcopy
from scipy.stats import randint


class BaseDistribution():
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
    def fit(cls, values):
        param = cls.dist_class.fit(values[~np.isnan(values)])
        return cls(*param)

    def to_dict(self):
        properties = deepcopy(self.par)
        properties["name"] = type(self).__name__
        return properties

    def draw(self):
        return self.dist.rvs()


class UniformDistribution(BaseDistribution):
    dist_class = uniform

    def __init__(self, min_val, max_val):
        self.par = {"min_val": min_val, "max_val": max_val}
        self.dist = uniform(loc=self.min_val, scale=self.max_val-self.min_val)

    @classmethod
    def fit(cls, values):
        vals = values[~np.isnan(values)]
        delta = vals.max() - vals.min()
        return cls(vals.min()-1e-3*delta, vals.max()+1e-3*delta)


class NormalDistribution(BaseDistribution):
    dist_class = norm

    def __init__(self, mean, std_dev):
        self.par = {"mean": mean, "std_dev": std_dev}
        self.dist = norm(loc=mean, scale=std_dev)


class FloatDistribution():
    var_types = [UniformDistribution, NormalDistribution]

    @classmethod
    def from_dict(cls, meta_dict):
        meta_dict = deepcopy(meta_dict)
        name = meta_dict.pop("name")
        for var_type in FloatDistribution.var_types:
            if name == var_type.__name__:
                return var_type(**meta_dict)
        raise ValueError("Cannot find right class.")

    @classmethod
    def fit(cls, values):
        instances = [var_type.fit(values)
                     for var_type in FloatDistribution.var_types]
        print([inst.AIC(values) for inst in instances])
        i_min = np.argmin([inst.AIC(values) for inst in instances])
        return instances[i_min]


class DiscreteUniformDistribution(BaseDistribution):
    dist_class = randint

    def __init__(self, low, high):
        self.par = {"low": low, "high": high}
        print(type(low), type(high))
        self.dist = self.dist_class(low=low, high=high+1)

    @classmethod
    def fit(cls, values):
        param = {"low": np.min(values), "high": np.max(values)}
        return cls(**param)


class IntDistribution():
    var_types = [DiscreteUniformDistribution]

    @classmethod
    def fit(cls, values):
        return DiscreteUniformDistribution.fit(values)


class CategoricalDistribution():
    def __init__(self, cat_freq):
        self.cat_freq = cat_freq

    @classmethod
    def fit(cls, values):
        unq_vals, counts = np.unique(values, return_counts=True)
        return cls(dict(zip(unq_vals, counts)))

    def to_dict(self):
        cat_freq = deepcopy(self.cat_freq)
        cat_freq["name"] = type(self).__name__
        return cat_freq

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        p_vals = np.array([x for x in self.cat_freq.values()])
        p_vals = p_vals/np.sum(p_vals)
        return np.random.choice(np.array(list(self.cat_freq)), p=p_vals)

# class NumericalVar():
#     def __init__(self, values, extra_nan=0):
#         self.nan_frac = (np.sum(np.isnan(values)) + extra_nan)/(len(values) + extra_nan)
#         self.average = np.nanmean(values)
#         self.min = np.nanmin(values)
#         self.max = np.nanmax(values)
#         self.n_unique = len(np.unique(values[np.logical_not(np.isnan(values))]))
# 
#     @staticmethod
#     def from_series(series):
#         if series.dtype == "int64":
#             return IntegerVar(series.values)
#         assert series.dtype == "float64"
# 
#     def __str__(self):
#         return (f"average: {self.average}, min: {self.min}, max: {self.max}, nan_frac: {self.nan_frac}, "
#                 f"unique: {self.n_unique}")
# 
#     def synth(self, n):
#         return [self.synth_one() for _ in range(n)]
# 
# 
# class IntegerVar(NumericalVar):
#     dtype = "int"
# 
#     def synth_one(self):
#         if np.random.random() < self.nan_frac:
#             return np.nan
#         return np.random.randint(self.min, self.max+1)
# 
# 
# class FloatVar(NumericalVar):
#     dtype = "float"
# 
#     def synth_one(self):
#         if np.random.random() < self.nan_frac:
#             return np.nan
#         return self.min + np.random.rand()*(self.max-self.min)


class StringDistribution():
    def __init__(self, str_lengths, p_length, all_char_counts):
        self.str_lengths = str_lengths
        self.p_length = p_length
        self.all_char_counts = all_char_counts

    @classmethod
    def fit(cls, values):
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
        data = self.to_dict()
        return f"String variable: avg # of characters: {np.sum(data['str_lengths']*data['p_length'])} "

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "str_lengths": deepcopy(self.str_lengths),
            "p_length": deepcopy(self.p_length),
            "all_char_counts": deepcopy(self.all_char_counts),
        }

    def draw(self):
        cur_str = ""
        str_len = np.random.choice(self.str_lengths, p=self.p_length)
        for i_chr in range(str_len):
            char_choices, p_choices = self.all_char_counts[i_chr]
            cur_str += np.random.choice(char_choices, p=p_choices)
        return cur_str


class NanDistribution():
    def to_dict(self):
        return {"name": type(self).__class__}

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        return np.nan


# class StringVar():
#     dtype="str"
#     def __init__(self, values):
#         values = values.astype(str)
#         self.n_unique = len(np.unique(values))
#         self.id_var = (self.n_unique == len(values))
#         self.categories = np.unique(values)
#         self.n_nan = np.sum(values == "")
#         self.nan_frac = self.n_nan/len(values)
#         
# 
#     def __str__(self):
#         return f"unique: {self.n_unique}, id_var: {self.id_var}, n_nan: {self.n_nan}"
# 
#     def synth_one(self):
#         if np.random.random() < self.nan_frac:
#             return ""
#         cur_str = ""
#         str_len = np.random.choice(self.str_lengths, p=self.p_length)
#         for i_chr in range(str_len):
#             char_choices, p_choices = self.all_char_counts[i_chr]
#             cur_str += np.random.choice(char_choices, p=p_choices)
#         return cur_str
# 
#     def synth(self, n):
#         return [self.synth_one() for _ in range(n)]
# 
# 
# class NanVar():
#     dtype="nan"
#     def __str__(self):
#         return "NanVar"
# 
#     def synth(self, n):
#         return [np.nan for _ in range(n)]
# 
# class DateVar():
#     pass