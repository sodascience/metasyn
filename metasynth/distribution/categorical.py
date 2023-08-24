"""Module containing categorical distributions."""

from __future__ import annotations

from typing import Union
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from metasynth.distribution.base import metadist, BaseDistribution


@metadist(implements="core.multinoulli", var_type=["categorical", "discrete", "string"])
class MultinoulliDistribution(BaseDistribution):
    """Categorical distribution that stores category labels and probabilities.

    Parameters
    ----------
    labels: list of str
        List containing the label belonging to each category.
    probs: list of int
        List containing the probability of each category.
        Probabilities will be normalized, so frequencies are valid too.
    """

    def __init__(self, labels: Union[npt.NDArray[np.str_], list[str]],
                 probs: Union[npt.NDArray[np.float_], list[float]]):
        self.labels = np.array(labels)
        self.probs = np.array(probs)
        if not np.isclose(np.sum(self.probs), 1):
            if np.any(self.probs < 0):
                raise ValueError("Cannot create multinoulli distribution with probabilities < 0.")
            warnings.simplefilter("always")
            warnings.warn("Multinoulli probabilities do not add up to 1 "
                          f" ({np.sum(self.probs)}); they will be rescaled.")
        self.probs = self.probs/np.sum(self.probs)

    @classmethod
    def _fit(cls, values: pl.Series):
        labels, counts = np.unique(values, return_counts=True)
        probs = counts/np.sum(counts)
        return cls(labels, probs)

    def _param_dict(self):
        return {"labels": self.labels, "probs": self.probs}

    @classmethod
    def _param_schema(cls):
        return {
            "labels": {"type": "array", "uniqueItems": True},
            "probs": {"type": "array", "items": {"type": "number"}},
        }

    def draw(self):
        return np.random.choice(self.labels, p=self.probs)

    def information_criterion(self,
                              values: Union[pd.Series, pl.Series, npt.NDArray[np.str_]]
                              ) -> float:
        series = self._to_series(values)
        labels, counts = np.unique(series, return_counts=True)
        log_lik = 0.0
        pdict = dict(zip(self.labels, self.probs))
        if len(self.labels) > 1 and isinstance(self.labels[0], (int, np.int_)):
            # Do add-one smoothing, assume count of values is equal to one in initial fit.
            alpha = 1
            num_event = len(series)
            num_tot_cat = np.max(self.labels)-np.min(self.labels)+1
            num_zero_cat = num_tot_cat - len(self.labels)
            theta = (self.probs*num_event+alpha)/(num_event+num_tot_cat*alpha)
            theta_zero = alpha/(num_event+num_tot_cat*alpha)
            theta_sum = np.sum(theta) + num_zero_cat*theta_zero
            theta_dict = dict(zip(self.labels, theta))
            for lab, count in zip(labels, counts):
                cur_theta = theta_dict.get(lab, theta_zero)
                cur_prob = cur_theta/theta_sum
                log_lik += count * np.log(cur_prob)
            return 2*(2*len(self.probs)-1) - 2*log_lik

        for lab, count in zip(labels, counts):
            # account for missing values / missing categories
            # by setting default of .get to 1 (add log(1)=0 to log_lik)
            log_lik += count * np.log(pdict.get(lab, 1))

        return 2*(2*len(self.probs)-1) - 2 * log_lik

    @classmethod
    def default_distribution(cls):
        return cls(["a", "b", "c"], [0.1, 0.3, 0.6])
