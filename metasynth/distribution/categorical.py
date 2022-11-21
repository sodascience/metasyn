"""Module containing categorical distributions."""

from typing import Union

import pandas as pd
import numpy as np
import numpy.typing as npt
import polars as pl

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
        if np.sum(self.probs) != 1:
            self.probs = self.probs/np.sum(self.probs)

    @classmethod
    def _fit(cls, values: pl.Series):
        labels, counts = np.unique(values, return_counts=True)
        probs = counts/np.sum(counts)
        return cls(labels.astype(str), probs)

    @property
    def par_dict(self):
        """Get labels and probabilities as a dictionary."""
        return dict(zip(self.labels, self.probs))

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

    def information_criterion(self,
                              values: Union[pd.Series, pl.Series, npt.NDArray[np.str_]]
                              ) -> float:
        values_array = np.array(values, dtype=str)
        labels, counts = np.unique(values_array, return_counts=True)
        log_lik = 0.0
        pdict = self.par_dict
        for lab, count in zip(labels, counts):
            # account for missing values / missing categories
            # by setting default of .get to 1 (add log(1)=0 to log_lik)
            log_lik += count * np.log(pdict.get(lab, 1))
        return 2*(len(self.probs) - 1) - 2 * log_lik

    @classmethod
    def default_distribution(cls):
        return cls(["a", "b", "c"], [10, 4, 20])
