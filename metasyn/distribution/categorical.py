"""Module implementing categorical distributions."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import numpy.typing as npt
import polars as pl

from metasyn.distribution.base import BaseDistribution, metadist


@metadist(implements="core.multinoulli", var_type=["categorical", "discrete", "string"])
class MultinoulliDistribution(BaseDistribution):
    """Categorical distribution using labels and probabilities.

    This class represents a multinoulli (categorical) distribution.
    It is used in cases where there are multiple potential outcomes,
    each with a specified probability. The class stores the labels for each
    category and their corresponding probabilities.

    Parameters
    ----------
    labels : list of str
        The labels for each category in the distribution, representing
        the possible outcomes.
    probs : list of int
        The probabilities or frequencies of each category. These will be
        normalized internally.

    Examples
    --------
    >>> MultinoulliDistribution(labels=["a", "b", "b"], probs=[0.1, 0.3, 0.6])
    >>> MultinoulliDistribution(labels=[1, 3, 6], probs=[0.3, 0.4, 0.3])
    """

    def __init__(
        self,
        labels: Union[npt.NDArray[Union[np.str_, np.int_]], list[Union[str, int]]],
        probs: Union[npt.NDArray[np.double], list[float]]
    ):
        self.labels = np.array(labels)
        self.probs = np.array(probs)
        if np.any(self.probs < 0):
            raise ValueError("Cannot create multinoulli distribution with probabilities < 0.")

        if not np.isclose(np.sum(self.probs), 1):
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
                              values: Union[pl.Series, npt.NDArray]
                              ) -> float:
        series = self._to_series(values)
        labels, counts = np.unique(series, return_counts=True)
        log_lik = 0.0
        pdict = dict(zip(self.labels, self.probs))
        # Check type of variable and act accordingly.
        if len(self.labels) > 1 and isinstance(self.labels[0], (int, np.int_)):
            log_lik = self._log_like_int(series, labels, counts)
            # See docstring of log_like_int method.
            n_parameters = len(self.probs)-1 + len(self.probs)
        else:
            for lab, count in zip(labels, counts):
                # account for missing values / missing categories
                # by setting default of .get to 1 (add log(1)=0 to log_lik)
                log_lik += count * np.log(pdict.get(lab, 1))
            n_parameters = len(self.probs)-1

        return np.log(len(series))*n_parameters - 2*log_lik

    def _log_like_int(
            self,
            series: pl.Series,
            labels: npt.NDArray[np.int_],
            counts: npt.NDArray[np.int_],
            alpha: float = 1.0,
            ) -> float:
        """Information criterion for the integer variant of the multinoulli distribution.

        This information criterion is a little bit different from the BIC, which is why
        the method is split into its own method. The reason is that with the default IC,
        the multinomial distribution is unfairly favored compared to the uniform distribution
        in particular (but really any discrete distribution). Assume for instance that we have a
        uniform distribution [0, 1000], with 10 values, then the log likelihood of the uniform
        distribution would be 10*log(1/1000), while the multinoulli has 10*log(1/10), which is
        much higher. To solve this problem we use additive smoothing:

        https://en.wikipedia.org/wiki/Additive_smoothing

        with parameter alpha = 1. This effectively says that all values in between the observed
        values have a probability of 1/N, where N is the number of total values. However, to
        prevent confusion (and potentially improve the synthesis) is draw from these unobserved
        values. That is why the information criterion for integers is not a true BIC, but a pseudo
        information criterion.

        The number of parameters is also in discussion, since we could imagine another multinoulli
        distribution with equal probs, which would have 'no' parameters. We could also see the
        multinoulli distribution as a dirac delta distribution with probability prefactors. Then it
        is easy to argue that the position of the dirac delta in effect is also a parameter. Thus,
        we choose to add an extra len(self.labels) term to keep the choice of distribution
        consistent.
        """
        # series = self._to_series(values)
        # labels, counts = np.unique(series, return_counts=True)
        log_lik = 0.0
        if len(self.labels) <= 1:
            return 0
        # Do additive smoothing, assume count of values is equal to that value in initial fit.
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
        return log_lik

    @classmethod
    def default_distribution(cls):
        return cls(["a", "b", "c"], [0.1, 0.3, 0.6])
