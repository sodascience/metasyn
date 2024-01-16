"""Module with privacy classes to be used for creating GMF files."""

from abc import ABC, abstractmethod
from typing import Optional, Type, Union

try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore

from metasyn.distribution.base import BaseDistribution


class BasePrivacy(ABC):
    """Base class for privacy level.

    Derived classes should at least set the class variable
    name and implement the to_dict method.
    """

    name = "unknown_privacy"

    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Create a dictionary that gives the privacy type, and parameters."""
        return {
            "type": self.name,
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
        return {}


class BasicPrivacy(BasePrivacy):
    """No privacy class, which uses statistically optimal distributions."""

    name = "none"

    def to_dict(self) -> dict:
        return BasePrivacy.to_dict(self)


def get_privacy(name: str, parameters: Optional[dict] = None):
    parameters = parameters if parameters is not None else {}
    for entry in entry_points(group="metasyn.privacy"):
        if name == entry.name:
            return entry.load()(**parameters)
    raise KeyError(f"Unknown privacy type with name '{name}'. "
                    "Ensure that you have installed the privacy package.")
