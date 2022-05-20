from copy import deepcopy
import re
import string
import random

import numpy as np

from metasynth.distribution.base import BaseDistribution


def get_prop_solved(values, new_values):
    n_char = np.sum([len(val) for val in values])
    n_new_char = np.sum([len(val) for val in new_values])
    return (n_char-n_new_char)/n_char


class MetaRegex():
    def __init__(self, re_list):
        self.re_list = re_list

    @classmethod
    def fit(cls, values):
        n_char = np.sum([len(val) for val in values])
        if n_char == 0:
            return cls([])
        
        best_solution = None
        for re_class in [REDigit, REOptionChar, REAlphaNumeric, RELetters, RELettersLower, RELettersUpper]:
            new_values, regexer = re_class.fit(values)
            prop_solved = get_prop_solved(values, new_values)
            cur_IC = regexer.IC(prop_solved)
            if best_solution is None or cur_IC > best_solution["IC"]:
                best_solution = {
                    "IC": cur_IC,
                    "re": regexer,
                    "values": new_values,
                }
        return cls([best_solution["re"]]) + cls.fit(best_solution["values"])

    def __add__(self, other):
        return MetaRegex(self.re_list + other.re_list)

    def draw(self):
        cur_str = ""
        for rex in self.re_list:
            cur_str += rex.draw()
        return cur_str

    @property
    def re(self):
        return "".join([x.re for x in self.re_list])

class BaseRE():
    def IC(self, prop_solved):
        ic_add = self.n_param + self.log_options
        return -ic_add/(prop_solved+1e-8)


class REBaseRex(BaseRE):
    regex_str = None
    def __init__(self, min_digit, max_digit):
        self.min_digit = min_digit
        self.max_digit = max_digit

    @property
    def n_param(self):
        return 1


    @classmethod
    def fit(cls, values):
        r = re.compile(cls.regex_str)
        new_values = []

        min_match = 99999
        max_match = 0
        for val in values:
            match = r.search(val)
            if match is None:
                n_match = 0
            else:
                n_match = match.span()[1]
            min_match = min(n_match, min_match)
            max_match = max(n_match, max_match)
            new_values.append(val[n_match:])
        return new_values, cls(min_match, max_match)


class REDigit(REBaseRex):
    regex_str = r"^\d{1,}"
    @property
    def log_options(self):
        return self.max_digit*np.log(10)

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([str(np.random.randint(0, 10)) for _ in range(n_digit)])

    @property
    def re(self):
        return "\d{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class REAlphaNumeric(REBaseRex):
    regex_str = r"^\w{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters+string.digits))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters+string.digits) for _ in range(n_digit)])

    @property
    def re(self):
        return "\w{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class RELetters(REBaseRex):
    regex_str = r"^[a-zA-Z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_letters))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_letters) for _ in range(n_digit)])

    @property
    def re(self):
        return "[a-zA-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class RELettersLower(REBaseRex):
    regex_str = r"^[a-z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_lowercase))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(n_digit)])

    @property
    def re(self):
        return "[a-z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class RELettersUpper(REBaseRex):
    regex_str = r"^[A-Z]{1,}"

    @property
    def log_options(self):
        return self.max_digit*np.log(len(string.ascii_uppercase))

    def draw(self):
        n_digit = np.random.randint(self.min_digit, self.max_digit+1)
        return "".join([random.choice(string.ascii_uppercase) for _ in range(n_digit)])

    @property
    def re(self):
        return "[A-Z]{"+str(self.min_digit)+","+str(self.max_digit)+"}"


class REOptionChar(BaseRE):
    def __init__(self, character_selection):
        self.character_selection = character_selection

    def draw(self):
        return np.random.choice(self.character_selection)

    @property
    def n_param(self):
        return len(self.character_selection)

    @property
    def log_options(self):
        return np.log(len(self.character_selection))

    @classmethod
    def fit(cls, values):
        first_chars = set([v[0] if len(v) > 0 else "" for v in values])
        new_values = [v[1:] for v in values]
        return new_values, cls(list(first_chars))

    @property
    def re(self):
        return "["+"".join(self.character_selection)+"]"

class StringRgxDistribution(BaseDistribution):
    def __init__(self):
        pass

    @classmethod
    def _fit(cls, values):
        values = values.astype(str)

class StringFreqDistribution(BaseDistribution):
    """String distribution that computes the frequency of characters.

    For the values supplied to the fit function, compute the string
    length distribution. Then for each character position compute the
    distribution of the characters.

    When drawing from the distribution, first draw a random string length
    from the distribution,
    and then for each position of the string, draw a character from the
    distribution for that position.

    It works particularly well with very strongly formatted strings, such as
    C25, A38, etc. For natural language, do not expect this to work
    particularly well.

    Parameters
    ----------
    str_lengths: array_like of int
        All string lengths available in the dataset.
    p_length: array_like of float
        Probability of each of the `str_lengths`. Has the same size.
    all_char_counts: list of tuple
        For each character position a tuple of length two should be supplied:
        * Available characters
        * Probability that those characters are selected.
    """

    def __init__(self, str_lengths, p_length, all_char_counts):
        self.str_lengths = str_lengths
        self.p_length = p_length
        self.all_char_counts = all_char_counts

    @classmethod
    def _fit(cls, values):
        values = values.astype(str)
        str_lengths, str_len_counts = np.unique([len(x) for x in values],
                                                return_counts=True)
        p_length = str_len_counts/len(values)
        all_char_counts = []
        for i_chr in range(np.max(str_lengths)):
            cur_chars, cur_char_counts = np.unique(
                [x[i_chr] for x in values if len(x) > i_chr],
                return_counts=True)
            all_char_counts.append((cur_chars,
                                    cur_char_counts/np.sum(cur_char_counts)))
        return cls(str_lengths, p_length, all_char_counts)

    def __str__(self):
        avg_len = np.sum([self.str_lengths[i]*self.p_length[i]
                          for i in range(len(self.str_lengths))])
        return f"String variable: avg # of characters: {avg_len} "

    def to_dict(self):
        return {
            "name": type(self).__name__,
            "parameters": {
                "str_lengths": deepcopy(self.str_lengths),
                "p_length": deepcopy(self.p_length),
                "all_char_counts": deepcopy(self.all_char_counts),
            }
        }

    def draw(self):
        cur_str = ""
        str_len = np.random.choice(self.str_lengths, p=self.p_length)
        for i_chr in range(str_len):
            char_choices, p_choices = self.all_char_counts[i_chr]
            cur_str += np.random.choice(char_choices, p=p_choices)
        return cur_str
