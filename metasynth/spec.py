from typing import Optional, Union

import polars as pl

from metasynth.provider import BaseDistribution
from metasynth.privacy import BasePrivacy


class VariableSpec:
    AVAILABLE_OPTIONS = ['distribution', 'unique', 'description', 'privacy', 'prop_missing']

    def __init__(self, spec):
        self._spec = spec

    def __setattr__(self, key, value):
        if key in VariableSpec.AVAILABLE_OPTIONS:
            self._spec[key] = value
        elif key == "_spec":
            super().__setattr__(key, value)
        else:
            raise AttributeError(
                f"{key} is not a valid attribute. It must be one of {', '.join(VariableSpec.AVAILABLE_OPTIONS)}.")

    def __getattr__(self, key):
        if key in VariableSpec.AVAILABLE_OPTIONS:
            return self._spec.get(key, None)
        else:
            return super().__getattribute__(key)

    def pop(self, key, default=None):
        return self._spec.pop(key, default)

class MetaFrameSpec:
    def __init__(self, df: pl.DataFrame):
        self._df = df
        self._specs = {var: VariableSpec({}) for var in df.columns}
        for column in df.columns:
            setattr(self, column, VariableSpec({}))

    def __getattr__(self, var_name: str):
        if var_name not in self._specs:
            raise AttributeError(f"Column '{var_name}' does not exist in DataFrame")
        return self._specs[var_name]

    def __getitem__(self, var_name: str):
        return self.__getattr__(var_name)

    def to_dict(self):
        return {key: value._spec for key, value in self._specs.items()}
