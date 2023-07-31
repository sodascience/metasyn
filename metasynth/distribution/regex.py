from __future__ import annotations

from typing import Union

from regexmodel import RegexModel

from metasynth.distribution.base import metadist, BaseDistribution


@metadist(implements="core.regex", var_type="string", version="2.0")
class RegexDistribution(BaseDistribution):
    def __init__(self, regex_data: Union[str, list[dict], RegexModel]):
        self.regex_model = RegexModel(regex_data)

    @classmethod
    def _fit(cls, values, count_thres: int = 3, method: str = "fast"):
        return cls(RegexModel.fit(values, count_thres=count_thres, method=method))

    def draw(self):
        return self.regex_model.draw()

    def _param_dict(self):
        return {"regex_data": self.regex_model.serialize()}

    def __repr__(self):
        return str([link.__repr__() for link in self.regex_model.root_links])

    @classmethod
    def _param_schema(cls):
        return {
            "regex": {
                "type": "string"
            },
            "counts": {
                "type": "array"
            }
        }

    @classmethod
    def default_distribution(cls):
        return cls(r"[ABC][0-9]{3,4}")


@metadist(implements="core.unique_regex", var_type="string", is_unique=True)
class UniqueRegexDistribution(RegexDistribution):
    """Unique variant of the regex distribution.

    Same as the normal regex distribution, but checks whether a key
    has already been used.

    Parameters
    ----------
    re_list: list of BaseRegexElement
        List of basic regex elements in the order that they occur.
    """

    def __init__(self, regex_data: Union[str, list]):
        super().__init__(regex_data)
        self.key_set: set[str] = set()

    def draw_reset(self):
        self.key_set = set()

    def draw(self) -> str:
        n_try = 0
        while n_try < 1e5:
            new_val = super().draw()
            if new_val not in self.key_set:
                self.key_set.add(new_val)
                return new_val
            n_try += 1
        raise ValueError("Failed to draw unique string after 100.000 tries.")

    def information_criterion(self, values):
        return 99999
