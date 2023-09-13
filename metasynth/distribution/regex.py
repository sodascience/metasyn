"""Distribution for structured strings, using regexes."""
from __future__ import annotations

from typing import Union

from regexmodel import RegexModel

from metasynth.distribution.base import metadist, BaseDistribution, UniqueDistributionMixin


@metadist(implements="core.regex", var_type="string", version="2.0")
class RegexDistribution(BaseDistribution):
    """Distribution for structured strings.

    Examples that this distribution should work reasonably for are: email,
    ID's, telephone numbers, ip addresses, etc.

    Valid regexes are ([a][b]|[c][d]), or [A-Z][0-9]{2,3}. During the fit, the probabilities
    of each of the branches in the regex is also recorded so that synthetic values can be
    drawn from a better distribution.

    Parameters
    ----------
    regex_data:
        A valid input for the regex model. The two main ones is a serialized version
        of the model, and the other is a regex that satisfies falls within the language
        of the regexmodel package (which is a small subset of the python re package).
    """
    def __init__(self, regex_data: Union[str, list[dict], RegexModel]):
        self.regex_model = RegexModel(regex_data)

    @classmethod
    def _fit(cls, values, count_thres: int = 3, method: str = "auto"):
        """Fit a regex to structured strings.

        Arguments
        ---------
        values:
            Values to be fitted (pl.Series).
        count_thres:
            Threshold for regex elements, so that a regex element can only be used if
            the number of values satisfying said element is higher than the threshold.
        method:
            Method for fitting the regex model. Possible values are ["accurate", "fast", "auto"]
            The "auto" method switches between the "accurate" and "fast" methods depending on
            the number of characters (fast if #char > 10000) in the series.
        """
        if method == "auto":
            if values.str.lengths().sum() > 10000:
                method = "fast"
            else:
                method = "accurate"
        return cls(RegexModel.fit(values, count_thres=count_thres, method=method))

    def draw(self):
        return self.regex_model.draw()

    def _param_dict(self):
        return {"regex_data": self.regex_model.serialize()}

    @property
    def _params_formatted(self):
        return f"\t- regex: {self.regex_model.regex}"

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
class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):
    """Unique variant of the regex distribution."""
