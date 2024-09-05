"""Module for distribution and variable specifications."""
from __future__ import annotations

# from metasyn.util import VarSpec
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from metasyn.distribution.base import BaseDistribution
from metasyn.privacy import BasePrivacy, BasicPrivacy, get_privacy


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
    distribution: Optional[BaseDistribution] = None

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
                                             DistributionSpec, str]],
              unique: Optional[bool] = None,
              ) -> DistributionSpec:
        """Create a DistributionSpec instance from a variety of inputs.

        Parameters
        ----------
        dist_spec:
            Specification for the distribution in several types.
        unique:
            Whether the distribution is unique. This is only taken into account
            if dist_spec is None or a string.

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
            return cls(**dist_dict, distribution=dist_spec)
        if isinstance(dist_spec, str):
            return cls(implements=dist_spec, unique=unique)
        if dist_spec is None:
            return cls(unique=unique)
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

    def get_creation_method(self, privacy: BasePrivacy) -> dict:
        """Create a dictionary on how the distribution was created.

        Parameters
        ----------
        privacy
            Privacy object with which the dictionary is being created.

        Returns
        -------
            Dictionary containing all the non-default settings for the creation method.
        """
        ret_dict: dict[str, Any] = {"created_by": "metasyn"}
        for var in ["implements", "unique", "parameters", "version"]:
            if getattr(self, var) is not None:
                ret_dict[var] = getattr(self, var)
        if len(self.fit_kwargs) > 0:
            ret_dict["fit_kwargs"] = self.fit_kwargs
        if not isinstance(privacy, BasicPrivacy):
            ret_dict["privacy"] = privacy.to_dict()
        return ret_dict


class VarSpec():  # pylint: disable=too-few-public-methods
    """Data class for storing the specifications for variables.

    Parameters
    ----------
    name:
        Name of the variable/column.
    distribution, optional:
        Distribution to use for fitting/finding the distribution.
        Leave at None to allow metasyn to find the most suitable distribution
        automatically.

        >>> # Use normal distribution
        >>> distribution="normal"
        >>> # Use normal distribution with mean 0, standard deviation 1
        >>> distribution=NormalDistribution(0, 1)

    unique, optional:
        To set a column to be unique/key.
        This is only available for the integer and string datatypes. Setting a variable
        to unique ensures that the synthetic values generated for this variable are unique.
        This is useful for ID or primary key variables, for example. The parameter...
        is ignored when the distribution is set manually. For example:
        {"unique": True}, which sets the variable to be unique or {"unique": False} which
        forces the variable to be not unique. If the uniqueness is not specified, it is
        assumed to be not unique, but gives a warning if metasyn thinks it should be.
    privacy, optional:
        Set the privacy level for a variable, e.g.: DifferentialPrivacy(epsilon=10).
    prop_missing, optional:
        Proportion of missing values for a variable.
    description, optional:
        Set the description of a variable.
    data_free, optional:
        Whether this variable/column is to be generated from scratch or from an existing column
        in the dataframe.
    var_type, optional:
        Manually set the variable type of the columns (used mainly for data_free columns).
    """

    def __init__(
            self,
            name: str,
            distribution: Optional[Union[dict, type[BaseDistribution], BaseDistribution,
                                                DistributionSpec, str]] = None,
            unique=None,
            privacy: Optional[BasePrivacy] = None,
            prop_missing: Optional[float] = None,
            description: Optional[str] = None,
            data_free: bool = False,
            var_type: Optional[str] = None):

        self.name = name
        self.dist_spec = DistributionSpec.parse(distribution, unique)
        self.privacy = privacy
        self.prop_missing = prop_missing
        self.description = description
        self.data_free = data_free
        self.var_type = var_type
        self.__post_init__()

    def __post_init__(self):
        # Convert the the privacy attribute if it is a dictionary.
        if isinstance(self.privacy, dict):
            self.privacy = get_privacy(**self.privacy)
        if self.data_free and not self.dist_spec.fully_specified:
            raise ValueError("Error creating variable specification: data free variable should have"
                            f" 'implements' and 'parameters'. {self}")

    @classmethod
    def from_dict(cls, var_dict: dict) -> VarSpec:
        """Create a variable specification from a dictionary.

        Parameters
        ----------
        var_dict
            Dictionary to parse the specification from.

        Returns
        -------
            A VarSpec instance.
        """
        return cls(**var_dict)
