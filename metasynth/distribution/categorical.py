"""Module containing categorical distributions."""

from typing import Sequence

import numpy as np
import numpy.typing as npt


from metasynth.distribution.base import CategoricalDistribution


class CatFreqDistribution(CategoricalDistribution):
    """Categorical distribution that stores category frequencies.

    Parameters
    ----------
    cat_freq: dict of int
        Dictionary containing the frequency of each category.
        The frequencies do not need to be normalized.
    """

    aliases = ["cat_freq"]

    def __init__(self, categories: npt.NDArray[np.str_],
                 counts: npt.NDArray[np.int_]):
        self.counts = counts
        self.categories = categories
        self.p_vals: npt.NDArray[np.float_] = np.array(list(counts))
        self.p_vals = self.p_vals/np.sum(self.p_vals)

    @classmethod
    def _fit(cls, values: Sequence[str]):
        categories, counts = np.unique(values, return_counts=True)
        return cls(categories.astype(str), counts)

    def to_dict(self):
        dist_dict = {}
        dist_dict["name"] = type(self).__name__
        dist_dict["parameters"] = {"categories": self.categories,
                                   "counts": self.counts}
        return dist_dict

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        return str(np.random.choice(self.categories, p=self.p_vals))
