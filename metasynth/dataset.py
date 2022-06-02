"""Conversion of pandas dataframes to MetaSynth datasets."""   # pylint: disable=invalid-name

import json

import numpy as np
import pandas as pd

from metasynth.var import MetaVar


class MetaDataset():
    """MetaSynth dataset consisting of variables.

    The MetaSynth dataset structure that is most easily created from
    a pandas dataset with the from_dataframe class method.

    Parameters
    ----------
    meta_vars: list of MetaVar
        List of variables representing columns in a dataframe.
    n_rows: int
        Number of rows in the original dataframe.
    """

    def __init__(self, meta_vars, n_rows=None):
        self.meta_vars = meta_vars
        self.n_rows = n_rows

    @property
    def n_columns(self):
        """int: Number of columns of the original dataframe."""
        return len(self.meta_vars)

    @classmethod
    def from_dataframe(cls, df, distribution=None, unique=None):
        """Create dataset from a Pandas dataframe.

        The pandas dataframe should be formatted already with the correct
        datatypes.

        Parameters
        ----------
        df: pandas.Dataframe
            Pandas dataframe with the correct column dtypes.
        distribution: dict of str or BaseDistribution, optional
            A dictionary that has keys that are column names and values that
            denote distributions, either with a string that gives one of their
            aliases. Or an actually fitted BaseDistribution.
        unique: dict of bool, optional
            A dictionary that allows specific columns to be set to be unique.
            This is only available for the integer and string datatypes. The parameter
            is ignored when the distribution is set manually.

        Returns
        -------
        MetaDataset:
            Initialized MetaSynth dataset.
        """
        if distribution is None:
            distribution = {}

        if unique is None:
            unique = {}

        all_vars = []
        for col_name in list(df):
            series = df[col_name]
            dist = distribution.get(col_name, None)
            unq = unique.get(col_name, None)
            var = MetaVar.detect(series)
            var.fit(dist, unique=unq)

            all_vars.append(var)

        return cls(all_vars, len(df))

    def to_dict(self):
        """Create dictionary with the properties for recreation."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "vars": [var.to_dict() for var in self.meta_vars],
        }

    def __getitem__(self, key):
        """Return meta var either by variable name or index."""
        if isinstance(key, int):
            return self.meta_vars[key]
        if isinstance(key, str):
            for var in self.meta_vars:
                if var.name == key:
                    return var
            raise KeyError(f"Cannot find variable '{key}'")
        raise TypeError(f"Cannot get item for key '{key}'")

    def __str__(self):
        """Create a readable string that shows the variables."""
        cur_str = "# Rows: "+str(self.n_rows)+"\n"
        cur_str += "# Columns: "+str(self.n_columns)+"\n"
        for var in self.meta_vars:
            cur_str += "\n"+str(var)+"\n"
        return cur_str

    def to_json(self, fp):
        """Write the MetaSynth dataset to a JSON file.

        Parameters
        ----------
        fp: str or pathlib.Path
            File to write the dataset to.
        """
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(_jsonify(self.to_dict()), f)

    @classmethod
    def from_json(cls, fp):
        """Read a MetaSynth dataset from a JSON file.

        Parameters
        ----------
        fp: str or pathlib.Path
            Path to read the data from.

        Returns
        -------
        MetaDataset:
            A restored metadataset from the file.
        """
        with open(fp, "r", encoding="utf-8") as f:
            self_dict = json.load(f)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows)

    def synthesize(self, n):
        """Create a synthetic pandas dataframe.

        Parameters
        ----------
        n: int
            Number of rows to generate.

        Returns
        -------
        pandas.DataFrame:
            Dataframe with the synthetic data.
        """ 
        synth_dict = {var.name: var.draw_series(n) for var in self.meta_vars}
        return pd.DataFrame(synth_dict)


def _jsonify(data):
    if isinstance(data, (list, tuple)):
        return [_jsonify(d) for d in data]
    if isinstance(data, dict):
        return {key: _jsonify(value) for key, value in data.items()}

    if isinstance(data, (np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    if isinstance(data, np.ndarray):
        return _jsonify(data.tolist())
    return data
