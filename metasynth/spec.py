from typing import Optional, Union

import polars as pl

from metasynth.provider import BaseDistribution
from metasynth.privacy import BasePrivacy

AVAILABLE_TYPES = {
    "distribution": [BaseDistribution, type(BaseDistribution), str],
    "unique": [bool],
    "description": [str],
    "privacy": [BasePrivacy, type(BasePrivacy), str],
    "prop_missing": [float],
    "fit_kwargs": [dict]
}


def type_check(key, value, type_dict=AVAILABLE_TYPES):
    if not any(isinstance(value, allowed_type) for allowed_type in
               type_dict[key]) and value is not None:
        raise TypeError(
            f"Expected type(s): '"
            f"{', '.join([allowed_type.__name__ for allowed_type in type_dict[key]])}' for "
            f"attribute '{key}', got '{type(value).__name__}'")


class VariableSpec:
    """
    A class to represent specifications for a variable (column) in a dataframe.
    """

    def __init__(self):
        """
        Construct a new 'VariableSpec' object with all attributes set to None.
        """
        self.distribution = None
        self.unique = None
        self.description = None
        self.privacy = None
        self.prop_missing = None
        self.fit_kwargs = None

    def __setattr__(self, key, value):
        """
        Set an attribute to the 'VariableSpec' object. Validates the attribute name and its type.

        Parameters
        ----------
        key : str
            The name of the attribute.
        value : any type that matches the attribute's valid types
            The value of the attribute.

        Raises
        ----------
        TypeError
            If the attribute's value does not match its valid types.
        """
        type_check(key, value)
        super().__setattr__(key, value)

    def to_dict(self):
        """
        Convert the 'VariableSpec' object to a dictionary, excluding attributes with None values.

        Returns
        ----------
        dict
            A dictionary that represents the 'VariableSpec' object.
        """
        dict_repr = {
            'distribution': self.distribution,
            'unique': self.unique,
            'description': self.description,
            'privacy': self.privacy,
            'prop_missing': self.prop_missing,
            'fit_kwargs': self.fit_kwargs
        }
        return {key: value for key, value in dict_repr.items() if value is not None}


class MetaFrameSpec:
    """
       A class to represent specifications for all variables (columns) in a dataframe.

       Attributes
       ----------
       columns : dict
           A dictionary where keys are column names and values are VariableSpec objects.
    """

    def __init__(self, df: pl.DataFrame):
        """
               Construct a new 'MetaFrameSpec' object.

               Parameters
               ----------
               df : DataFrame
                   The dataframe from where column names are extracted.
        """
        self.columns = {col: VariableSpec() for col in df.columns}

    def __getitem__(self, item):
        """
              Return the VariableSpec object for a given column.

              Parameters
              ----------
              item : str
                  The column name.

              Returns
              ----------
              VariableSpec
                  The VariableSpec object for the column.
        """
        return self.columns[item]

    def __setitem__(self, key, value):
        """
               Set a VariableSpec object to a given column.

               Parameters
               ----------
               key : str
                   The column name.
               value : VariableSpec
                   The VariableSpec object to set.

               Raises
               ----------
               ValueError
                   If value is not a VariableSpec instance.
        """
        if not isinstance(value, VariableSpec):
            raise ValueError("Value must be a VariableSpec instance.")
        self.columns[key] = value

    def to_dict(self):
        """
               Convert the MetaFrameSpec object to a dictionary.

               Returns
               ----------
               dict
                   A dictionary where keys are column names and values are dictionary
                   representations of the VariableSpec objects.
        """
        return {col: spec.to_dict() for col, spec in self.columns.items()}