"""Module with privacy classes to be used for creating GMF files."""
from abc import ABC, abstractmethod
from typing import Optional, Type, Union

try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore

from metasyn.distribution.base import BaseDistribution
from metasyn.util import get_registry


class BasePrivacy(ABC):
    """Abstract base class for privacy levels.

    This class serves as a blueprint for privacy classes. Derived classes
    should at least set the class variable `name` and implement the `to_dict`
    method, which should return a dictionary that gives the privacy type and
    its parameters.
    """

    name = "unknown_privacy"

    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Create a dictionary that gives the privacy type, and parameters."""
        return {
            "name": self.name,
            "parameters": {},
        }

    def is_compatible(self, dist: Union[BaseDistribution, Type[BaseDistribution]]) -> bool:
        """Check whether the distribution has the same privacy.

        Arguments
        ---------
        dist:
            Distribution to check.
        """
        return dist.privacy == self.name

    @property
    def fit_kwargs(self):
        """Fitting arguments that need to be supplied to the distribution.

        For example epsilon in the case of differential privacy.
        """
        return self.to_dict()["parameters"]


class BasicPrivacy(BasePrivacy):
    """Class representing no privacy level.

    This class uses statistically optimal distributions. It inherits from the
    `BasePrivacy` class and sets the `name` attribute to "none".
    """

    name = "none"

    def to_dict(self) -> dict:
        return BasePrivacy.to_dict(self)


def get_privacy(name: str, parameters: Optional[dict] = None) -> BasePrivacy:
    """Create a new privacy object using a name and parameters.

    Parameters
    ----------
    name
        Name of the privacy type, use "none" for no specific type of privacy.
    parameters, optional
        The parameters for the privacy type. This could be the epsilon for differential
        privacy or partition_size for disclosure control, by default None.

    Returns
    -------
        A new instantiated object for privacy.

    Raises
    ------
    KeyError
        If the name of the privacy type cannot be found.
    """
    parameters = parameters if parameters is not None else {}
    for entry in entry_points(group="metasyn.privacy"):
        if name == entry.name:
            return entry.load()(**parameters)

    # Handle case where the plugin is not installed or is misspelled.
    registry = get_registry()
    available_plugins = {key: val for key, val in registry.items() if name in val["privacy"]}
    if len(available_plugins) > 0:
        avail = "\n".join(f"{key}: {val['url']}" for key, val in available_plugins.items())
        raise ImportError(f"No plugin is installed that provides '{name}' privacy. "
                          f"Available plugins that provide '{name}' privacy:\n\n{avail}")
    privacy_names = [entry.name for entry in entry_points(group="metasyn.privacy")]
    avail = "\n".join(f"<{key}> provides: {val['privacy']}\n\t{val['url']}"
                      for key, val in registry.items() if len(val['privacy']) > 0)
    raise ImportError(f"Unknown privacy type with name '{name}'. "
                       "Ensure that you have installed the correct privacy package"
                       " (and not misspelled it).\n"
                      f"Installed privacy types: {privacy_names}.\n\n"
                      f"List of plugins that provide privacy:\n\n{avail}\n")
