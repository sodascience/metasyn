"""Module containing categorical distributions."""

from typing import Sequence, Iterable

import numpy as np
import numpy.typing as npt


from metasynth.distribution.base import CategoricalDistribution


class MultinoulliDistribution(CategoricalDistribution):
    """Categorical distribution that stores category labels and probabilities.

    Parameters
    ----------
    labels: list of str
        List containing the label belonging to each category.
    probs: list of int
        List containing the probability of each category.
        Probabilities will be normalized, so frequencies are valid too.
    """

    aliases = ["MultinoulliDistribution", "categorical"]

    def __init__(self, labels: npt.NDArray[np.str_],
                 probs: npt.NDArray[np.float_]):
        self.labels = labels
        self.probs = probs
        if (np.sum(self.probs) != 1):
            self.probs = self.probs/np.sum(self.probs)

    @classmethod
    def _fit(cls, values: Sequence[str]):
        categories, counts = np.unique(values, return_counts=True)
        return cls(categories.astype(str), counts)

    def to_dict(self):
        dist_dict = {}
        dist_dict["name"] = type(self).__name__
        dist_dict["parameters"] = {"labels": self.labels,
                                   "probs": self.probs}
        return dist_dict

    def __str__(self):
        return str(self.to_dict())

    def draw(self):
        return str(np.random.choice(self.labels, p=self.probs))

    def information_criterion(self, values: Iterable) -> float:
        ll = 0.0
        for v in values:
            ll += np.log(self.probs[self.labels.index(v)])
        return 2*(len(self.probs) - 1) - 2 * ll

    @classmethod
    def _example_distribution(cls):
        return cls(["a", "b", "c"], [10, 4, 20])
