from typing import Optional, Union

import polars as pl

from metasynth.provider import BaseDistribution
from metasynth.privacy import BasePrivacy


class ColumnSpec:
    AVAILABLE_OPTIONS = ['distribution', 'unique', 'description', 'privacy', 'prop_missing']

    def __init__(self, spec):
        self._spec = spec

    def __setattr__(self, key, value):
        if key in ColumnSpec.AVAILABLE_OPTIONS:
            self._spec[key] = value
        elif key == "_spec":
            super().__setattr__(key, value)
        else:
            raise AttributeError(
                f"{key} is not a valid attribute. It must be one of {', '.join(ColumnSpec.AVAILABLE_OPTIONS)}.")

    def __getattr__(self, key):
        if key in ColumnSpec.AVAILABLE_OPTIONS:
            return self._spec.get(key, None)
        else:
            return super().__getattribute__(key)

    def pop(self, key, default=None):
        return self._spec.pop(key, default)

class MetaVarSpec:
    def __init__(self, df: pl.DataFrame):
        self._df = df
        self._specs = {col: ColumnSpec({}) for col in df.columns}

    def __getattr__(self, col_name: str):
        if col_name not in self._df.columns:
            raise AttributeError(f"Column '{col_name}' does not exist in DataFrame")
        return self._specs[col_name]

    def __getitem__(self, col_name: str):
        return self.__getattr__(col_name)

    def to_dict(self):
        return {k: v._spec for k, v in self._specs.items()}

# class MetaVarSpec:
#     def __init__(self, df: pl.DataFrame):
#         self._df = df
#         self._specs = {}
#
#         for col in df.columns:
#             self.add_var(col)
#
#     def add_var(self, col_name: str,
#                 distribution: Optional[Union[str, BaseDistribution]] = None,
#                 unique: Optional[bool] = None,
#                 description: Optional[str] = None,
#                 privacy: Optional[BasePrivacy] = None,
#                 prop_missing: Optional[float] = None):
#
#         spec = {}
#         if distribution is not None:
#             spec["distribution"] = distribution
#         if unique is not None:
#             spec["unique"] = unique
#         if description is not None:
#             spec["description"] = description
#         if privacy is not None:
#             spec["privacy"] = privacy
#         if prop_missing is not None:
#             spec["prop_missing"] = prop_missing
#
#         self._specs[col_name] = ColumnSpec(spec)
#
#     def __getattr__(self, col_name: str):
#         if col_name not in self._df.columns:
#             raise AttributeError(f"Column '{col_name}' does not exist in DataFrame")
#
#         # if col_name not in self._specs:
#         #     self.add_var(col_name)
#
#         return self._specs[col_name]
