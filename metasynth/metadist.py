"""
Module containing all MetaDistributions.

The MetaDistributions can find and fit
multiple distributions with the correct types.
"""

from abc import ABC
from copy import deepcopy

import numpy as np

from metasynth.distribution import UniformDistribution, NormalDistribution
from metasynth.distribution import DiscreteUniformDistribution, CatFreqDistribution
from metasynth.distribution import StringFreqDistribution


class MetaDistribution(ABC):
    """Abstract class to find and restore distributions.

    Each supported type has its own derived class. The purpose of each class
    at the moment is to keep track of which distributions apply to it. None
    of them need to be instantiated, so in that sense they are abstract.
    """

    dist_types = []

    @classmethod
    def from_dict(cls, meta_dict):
        """Restore/create distribution from dictionary.

        Parameters
        ----------
        meta_dict: dict
            Dictionary with the name and parameters of the distribution.

        Returns
        -------
        BaseDistribution:
            Distribution with the stored parameters.
        """
        meta_dict = deepcopy(meta_dict)
        name = meta_dict.pop("name")
        for dist_type in cls.dist_types:
            if name == dist_type.__name__:
                return dist_type(**meta_dict["parameters"])
        raise ValueError("Cannot find right class.")

    @classmethod
    def fit(cls, values):
        """Fit distribution to values.

        Find the distribution that is most suitable for the supplied data
        values, by fitting all available distributions and determining the
        quality of the fit using the AIC.

        Currently, most types have only a single distribution to chose from,
        but this will change.

        Arguments
        ---------
        values: array_like
            Values to fit the distributions to.

        Returns
        -------
        BaseDistribution:
            Distribution that fits the data the best.
        """
        instances = [dist_type.fit(values)
                     for dist_type in cls.dist_types]
        i_min = np.argmin([inst.AIC(values) for inst in instances])
        return instances[i_min]


class FloatDistribution(MetaDistribution):
    """Meta class for floating point distributions."""
    dist_types = [UniformDistribution, NormalDistribution]


class IntDistribution(MetaDistribution):
    """Meta class for integer distributions."""
    dist_types = [DiscreteUniformDistribution]


class CategoricalDistribution(MetaDistribution):
    """Meta class for categorical distributions."""
    dist_types = [CatFreqDistribution]


class StringDistribution(MetaDistribution):
    """Meta class for string distributions."""
    dist_types = [StringFreqDistribution]
