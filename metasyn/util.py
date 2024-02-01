"""Utility module for metasyn.

This module provides utility classes that are used across metasyn,
including classes for specifying distributions and storing variable
configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from metasyn.distribution.base import BaseDistribution
from metasyn.privacy import BasePrivacy, get_privacy


@dataclass
class DistributionSpec():
    """Specification that determines which distribution is selected.

    It has the following attributes:
    - implements: Which distribution is chosen.
    - unique: Whether the distribution should be unique.
    - parameters: The parameters of the distribution as defined by implements.
    - fit_kwargs: Fitting keyword arguments to be used while fitting the distribution.
    - version: Version of the distribution to fit.
    """

    implements: Optional[str] = None
    unique: Optional[bool] = None
    parameters: Optional[dict] = None
    fit_kwargs: dict = field(default_factory=dict)
    version: Optional[str] = None

    def __post_init__(self):
        if self.implements is None:
            if self.version is not None:
                raise ValueError("Cannot create DistributionSpec with attribute 'version' but "
                                 "without attribute 'implements'.")
            if self.parameters is not None:
                raise ValueError("Cannot create DistributionSpec with attribute 'parameters' but "
                                 "without attribute 'implements'.")
            if len(self.fit_kwargs) > 0:
                raise ValueError("Cannot create DistributionSpec with attribute 'fit_kwargs' that"
                                 " is not empty but without attribute 'implements'.")


    @classmethod
    def parse(cls, dist_spec: Optional[Union[dict, type[BaseDistribution], BaseDistribution,
                                             DistributionSpec, str]]
              ) -> DistributionSpec:
        """Create a DistributionSpec instance from a variety of inputs.

        Parameters
        ----------
        dist_spec
            Specification for the distribution in several types.

        Returns
        -------
            A instantiated version of the dist_spec that has the DistributionSpec type.

        Raises
        ------
        TypeError
            If the input has the wrong type and cannot be parsed.
        """
        if isinstance(dist_spec, BaseDistribution):
            dist_dict = {key: value for key, value in dist_spec.to_dict().items()
                         if key in ["implements", "version", "unique", "parameters"]}
            dist_dict["unique"] = dist_dict.pop("unique")
            return cls(**dist_dict)
        if isinstance(dist_spec, str):
            return cls(implements=dist_spec)
        if dist_spec is None:
            return cls()
        if isinstance(dist_spec, dict):
            return cls(**dist_spec)
        if isinstance(dist_spec, DistributionSpec):
            return dist_spec
        if issubclass(dist_spec, BaseDistribution):
            return cls(implements=dist_spec.implements, unique=dist_spec.unique)
        raise TypeError("Error parsing distribution specification of unknown type "
                        f"'{type(dist_spec)}' with value '{dist_spec}'")

    @property
    def fully_specified(self) -> bool:
        """Indicate whether the distribution is suitable for datafree creation.

        Returns
        -------
            A flag that indicates whether a distribution can be generated from the values
            that are specified (not None).
        """
        return self.implements is not None and self.parameters is not None

@dataclass
class VarConfig():
    """Data class for storing the configurations for variables.

    It contains the following attributes:
    - name: Name of the variable/column.
    - dist_spec: DistributionSpec object that determines the distribution.
    - privacy: Privacy object that determines which implementation can be used.
    - prop_missing: Proportion of missing values.
    - description: Description of the variable.
    - var_type: Type of the variable in question.
    """

    name: str
    dist_spec: DistributionSpec = field(default_factory=DistributionSpec)
    privacy: Optional[BasePrivacy] = None
    prop_missing: Optional[float] = None
    description: Optional[str] = None
    data_free: bool = False
    var_type: Optional[str] = None

    def __post_init__(self):
        # Convert the the privacy attribute if it is a dictionary.
        if isinstance(self.privacy, dict):
            self.privacy = get_privacy(**self.privacy)
        if self.data_free and not self.dist_spec.fully_specified:
            raise ValueError("Error creating variable specification: data free variable should have"
                            f" 'implements' and 'parameters'. {self}")

    @classmethod
    def from_dict(cls, var_dict: dict) -> VarConfig:
        """Create a variable configuration from a dictionary.

        Parameters
        ----------
        var_dict
            Dictionary to parse the configuration from.

        Returns
        -------
            A new VarConfig instance.
        """
        dist_spec = var_dict.pop("distribution", None)
        if dist_spec is None:
            return cls(**var_dict)
        return cls(**var_dict, dist_spec=DistributionSpec.parse(dist_spec))
