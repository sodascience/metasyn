"""
Module containing all MetaDistributions.

The MetaDistributions can find and fit
multiple distributions with the correct types.
"""

from abc import ABC
from copy import deepcopy

import numpy as np

from metasynth.distribution.util import get_dist_class


class MetaDistribution(ABC):
    """Abstract class to find and restore distributions.

    Each supported type has its own derived class. The purpose of each class
    at the moment is to keep track of which distributions apply to it. None
    of them need to be instantiated, so in that sense they are abstract.
    """

    # dist_types: List[Type[BaseDistribution]] = []

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
        dist_class, _ = get_dist_class(name)
        return dist_class(**meta_dict["parameters"])

    @classmethod
    def fit(cls, values, distributions, unique=None):
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
                     for dist_type in distributions
                     if unique is None or dist_type.is_unique == unique]
        instances = [inst for inst in instances if inst is not None]
        i_min = np.argmin([inst.information_criterion(values) for inst in instances])
        return instances[i_min]
