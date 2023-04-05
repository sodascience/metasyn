from abc import abstractmethod
from typing import Union, Type
from metasynth.distribution.base import BaseDistribution


class BasePrivacy():
    name = "unknown_privacy"

    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        return {
            "type": self.name,
            "parameters": {},
        }

    def is_compatible(self, dist: Union[BaseDistribution, Type[BaseDistribution]]) -> bool:
        return dist.privacy == self.name

    @property
    def fit_kwargs(self):
        return {}


class NoPrivacy(BasePrivacy):
    name = "none"

    def to_dict(self) -> dict:
        return BasePrivacy.to_dict(self)
