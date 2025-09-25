"""Regex distributions and fitters."""
from typing import Optional, Union

# from lingua._constant import LETTERS, PUNCTUATION
from regexmodel import NotFittedError, RegexModel

from metasyn.distribution.base import (
    BaseDistribution,
    BaseFitter,
    UniqueDistributionMixin,
    metadist,
    metafit,
)


@metadist(name="core.regex", var_type="string", version="2.0")
class RegexDistribution(BaseDistribution):
    """Structured string distribution using regex.

    Main implementation details in the regexmodel package:
    https://github.com/sodascience/regexmodel

    This distribution tries to create a regex that fits the variable. This
    regex also contains statistical information about the probabilities if the
    regex has multiple options (e.g. a|b). The regex is only a subset what is provided
    by the python re package. What is currently implemented:

    - Parentheses with multiple options and no modifiers, e.g. ([a]|[b]|[c])
    - Square brackets without negation, e.g. [abc]
    - Ranges [A-Z], [a-z], [0-9], but not subranges (e.g. [0-3])
    - Repetition quantifiers (curly brackets) with minimum and maximum [A-Z]{3,6}, but not [A-Z]{6}.

    When fitting the RegexDistribution using the fit method, pay attention to the
    count_thres and method parameters. By default these will be dynamic and take reasonable
    values for the input, but in some cases it can be important to set them manually.
    The count_thres parameter sets the minimum number of times a regex element needs to
    be used. So, if count_thres=2, and there is only one value starting with "a", then the regex
    will never start with "a". In effect, a higher value will provide more privacy, less
    utility and a faster fit. The other parameter "method" has a small effect on the accuracy
    of the regex, and a larger effect on the worst case time consumption for fitting.
    Set to "accurate" for the best result, and "fast" for the fastest result.

    Examples that this distribution should work reasonably for are: email,
    ID's, telephone numbers, ip addresses, etc.

    Parameters
    ----------
    regex_data:
        Valid inputs for the regex model are:
        - str: String with a regex (that falls within the specifications, see above).
        - dict: Serialized version of the regex model, as it is coming from a JSON file.
        - RegexModel: Initialized regex model.

    Examples
    --------
    >>> RegexDistribution(r"AB[0-9]{4}").draw()
    "AB8123"
    >>> RegexDistribution(r"(a|b|c)10)").draw()
    "b10"
    """

    def __init__(self, regex_data: Union[str, dict, RegexModel]):
        self.regex_model = RegexModel(regex_data)

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
    def default_distribution(cls, var_type=None) -> BaseDistribution: # noqa: ARG003
        return cls(r"[ABC][0-9]{3,4}")

    def information_criterion(self, values):
        mean_len = values.str.len_chars().mean()
        diff = ((values.str.len_chars() - mean_len)**2).mean()
        if mean_len > 3 and diff < 1:
            return -2
        return 0


@metafit(distribution=RegexDistribution, var_type="string", version="2.0")
class RegexFitter(BaseFitter):
    """Fitter for regex distribution."""

    distribution: type[RegexDistribution]

    def _fit(self, series, count_thres: Optional[int] = None, method: str = "auto"):
        """Fit a regex to structured strings.

        Arguments
        ---------
        series:
            Values to be fitted (pl.Series).
        count_thres:
            Threshold for regex elements, so that a regex element can only be used if
            the number of series satisfying said element is higher than the threshold.
        method:
            Method for fitting the regex model. Possible series are ["accurate", "fast", "auto"]
            The "auto" method switches between the "accurate" and "fast" methods depending on
            the number of characters (fast if #char > 10000) in the series.
        """
        if method == "auto":
            if series.str.len_chars().mean() > 10:
                method = "fast"
            else:
                method = "accurate"

        # Make count_thres ~= #series/100 up to 50 if in auto mode.
        if count_thres is None:
            count_thres = min(50, max(2, round(len(series)/50)))

        # Try to fit the series, if it cannot be fit, then use the default distribution.
        try:
            model = RegexModel.fit(series, count_thres=count_thres, method=method)
        except NotFittedError:
            return self.distribution.default_distribution()
        return self.distribution(model)



@metadist(name="core.regex", var_type="string", unique=True)
class UniqueRegexDistribution(UniqueDistributionMixin, RegexDistribution):
    """Unique variant of the regex distribution.

    See :class:`~RegexDistribution` for examples and explanation on this distribution.
    """

@metafit(distribution=UniqueRegexDistribution, var_type="string")
class UniqueRegexFitter(RegexFitter):
    """Fitter for unique regex distribution."""
