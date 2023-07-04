"""Module containing categorical distributions."""

from typing import Union
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from metasynth.distribution.base import (CategoricalDistribution,
                                         CoreDistribution)


class MultinoulliDistribution(CoreDistribution, CategoricalDistribution):
    """Categorical distribution that stores category labels and probabilities.

    Parameters
    ----------
    labels: list of str
        List containing the label belonging to each category.
    probs: list of int
        List containing the probability of each category.
        Probabilities will be normalized, so frequencies are valid too.
    """

    implements = "core.multinoulli"

    def __init__(self, labels: Union[npt.NDArray[np.str_], list[str]],
                 probs: Union[npt.NDArray[np.float_], list[float]]):
        self.labels = np.array(labels)
        self.probs = np.array(probs)
        if np.isclose(np.sum(self.probs), 1):
            if np.any(self.probs < 0):
                raise ValueError("Cannot create multinoulli distribution with probabilities < 0.")
            warnings.simplefilter("always")
            warnings.warn("Creating multinoulli distribution where probabilities do not add up to 1"
                          f" ({np.sum(self.probs)})")
            self.probs = self.probs/np.sum(self.probs)

    @classmethod
    def _fit(cls, values: pl.Series):
        labels, counts = np.unique(values, return_counts=True)
        probs = counts/np.sum(counts)
        return cls(labels.astype(str), probs)

    def _param_dict(self):
        return {"labels": self.labels, "probs": self.probs}

    @classmethod
    def _param_schema(cls):
        return {
            "labels": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "probs": {"type": "array", "items": {"type": "number"}},
        }

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
        pdict = dict(zip(self.labels, self.probs))
        for lab, count in zip(labels, counts):
            # account for missing values / missing categories
            # by setting default of .get to 1 (add log(1)=0 to log_lik)
            log_lik += count * np.log(pdict.get(lab, 1))
        return 2*(len(self.probs) - 1) - 2 * log_lik

    @classmethod
    def default_distribution(cls):
        return cls(["a", "b", "c"], [10, 4, 20])
