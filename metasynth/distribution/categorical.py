import numpy as np

from metasynth.distribution.base import BaseDistribution


class CatFreqDistribution(BaseDistribution):
    """Categorical distribution that stores category frequencies.

    Parameters
    ----------
    cat_freq: dict of int
        Dictionary containing the frequency of each category.
        The frequencies do not need to be normalized.
    """

    def __init__(self, cat_freq):
        self.cat_freq = cat_freq
        self.terms = list(cat_freq)
        self.p_vals = np.array(list(self.cat_freq.values()))
        self.p_vals = self.p_vals/np.sum(self.p_vals)

    @classmethod
    def _fit(cls, values):
        unq_vals, counts = np.unique(values, return_counts=True)
        return cls(dict(zip(unq_vals.astype(str), counts)))

    def to_dict(self):
        dist_dict = {}
        dist_dict["name"] = type(self).__name__
        dist_dict["parameters"] = {"cat_freq": self.cat_freq}
        return dist_dict

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        return str(np.random.choice(self.terms, p=self.p_vals))
