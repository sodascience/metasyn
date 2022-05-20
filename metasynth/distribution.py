"""Distributions for variables.

These distributions can be fit to datasets/series so that the synthesis is
somewhat realistic. The concept of distributions here is not only for
numerical data, but also for generating strings for example.
"""  # pylint: disable=invalid-name

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.stats import uniform, norm
from scipy.stats import randint


class BaseDistribution(ABC):
    """Abstract base class to define a distribution.

    All distributions should be derived from this class, and the following
    methods need to be implemented: _fit, draw, to_dict.
    """

    @classmethod
    def fit(cls, series):
        """Fit the distribution to the series.

        Parameters
        ----------
        series: pandas.Series
            Data to fit the distribution to.

        Returns
        -------
        BaseDistribution:
            Fitted distribution.
        """
        distribution = cls._fit(series.dropna())
        return distribution

    @classmethod
    @abstractmethod
    def _fit(cls, values):
        """See fit method, but does not need to deal with NA's."""

    @abstractmethod
    def draw(self):
        """Draw a random element from the fitted distribution."""

    def __str__(self):
        """Create a human readable string of the object."""
        return str(self.to_dict())

    @abstractmethod
    def to_dict(self):
        """Convert the distribution to a dictionary."""

    def AIC(self, values):  # pylint: disable=unused-argument, no-self-use
        """Get the AIC value for a particular set of values.

        TODO: Should probably rename, since this only makes (much) sense
        for numerical distributions.

        Parameters
        ----------
        values: array_like
            Values to determine the AIC value of.
        """
        return 0.0


class ScipyDistribution(BaseDistribution):
    """Base class for numerical Scipy distributions.

    This base class makes it easy to implement new numerical
    distributions. One could also use this base class for non-scipy
    distributions, in which case the distribution class should implement
    logpdf, rvs and fit methods.
    """

    @property
    def n_par(self):
        """int: Number of parameters for distribution."""
        return len(self.par)

    def __getattr__(self, attr):
        """Get attribute for easy access to parameters.

        Parameters
        ----------
        attr:
            Attribute to retrieve. If the attribute is a parameter
            name, then retrieve that parameter, otherwise use the default
            implementation for getting the attribute.

        Returns
        -------
        object:
            Parameter or attribute.
        """
        if attr != "par" and attr in self.par:
            return self.par[attr]
        return super().__getattr__(attr)  # pylint: disable=no-member

    @classmethod
    def _fit(cls, values):
        param = cls.dist_class.fit(values[~np.isnan(values)])
        return cls(*param)

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": deepcopy(self.par),
        }

    def draw(self):
        return self.dist.rvs()

    def AIC(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpdf(vals+1e-7))


class UniformDistribution(ScipyDistribution):
    """Uniform distribution for floating point type.

    This class implements the uniform distribution between a minimum
    and maximum.

    Parameters
    ----------
    min_val: float
        Lower bound for uniform distribution.
    max_val: float
        Upper bound for uniform distribution.
    """

    dist_class = uniform

    def __init__(self, min_val, max_val):
        self.par = {"min_val": min_val, "max_val": max_val}
        self.dist = uniform(loc=self.min_val, scale=self.max_val-self.min_val)

    @classmethod
    def _fit(cls, values):
        vals = values[~np.isnan(values)]
        delta = vals.max() - vals.min()
        return cls(vals.min()-1e-3*delta, vals.max()+1e-3*delta)


class NormalDistribution(ScipyDistribution):
    """Normal distribution for floating point type.

    This class implements the normal/gaussian distribution and takes
    the average and standard deviation as initialization input.

    Parameters
    ----------
    mean: float
        Mean of the normal distribution.

    std_dev: float
        Standard deviation of the normal distribution.
    """

    dist_class = norm

    def __init__(self, mean, std_dev):
        self.par = {"mean": mean, "std_dev": std_dev}
        self.dist = norm(loc=mean, scale=std_dev)


class DiscreteUniformDistribution(ScipyDistribution):
    """Integer uniform distribution.

    It differs from the floating point uniform distribution by
    being a discrete distribution instead.

    Parameters
    ----------
    low: int
        Lower bound (inclusive) of the uniform distribution.
    high: int
        Upper bound (exclusive) of the uniform distribution.
    """

    dist_class = randint

    def __init__(self, low, high):
        self.par = {"low": low, "high": high}
        self.dist = self.dist_class(low=low, high=high)

    def AIC(self, values):
        vals = values[~np.isnan(values)]
        return 2*self.n_par - 2*np.sum(self.dist.logpmf(vals.values.astype(int)+1e-7))

    @classmethod
    def _fit(cls, values):
        param = {"low": np.min(values), "high": np.max(values)+1}
        return cls(**param)


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


class StringFreqDistribution(BaseDistribution):
    """String distribution that computes the frequency of characters.

    For the values supplied to the fit function, compute the string
    length distribution. Then for each character position compute the
    distribution of the characters.

    When drawing from the distribution, first draw a random string length
    from the distribution,
    and then for each position of the string, draw a character from the
    distribution for that position.

    It works particularly well with very strongly formatted strings, such as
    C25, A38, etc. For natural language, do not expect this to work
    particularly well.

    Parameters
    ----------
    str_lengths: array_like of int
        All string lengths available in the dataset.
    p_length: array_like of float
        Probability of each of the `str_lengths`. Has the same size.
    all_char_counts: list of tuple
        For each character position a tuple of length two should be supplied:
        * Available characters
        * Probability that those characters are selected.
    """

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
