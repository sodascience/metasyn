import json

import numpy as np

from metasynth.var import MetaVar


class MetaDataset():
    def __init__(self, meta_vars, n_rows=None):
        self.meta_vars = meta_vars
        self.n_rows = n_rows

    @property
    def n_columns(self):
        return len(self.meta_vars)

    @classmethod
    def from_dataframe(cls, df):
        all_vars = [var for var in MetaVar.detect(df)]
        for var in all_vars:
            var.fit()
        return cls(all_vars, len(df))

    def to_dict(self):
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "vars": [var.to_dict() for var in self.meta_vars],
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.meta_vars[key]
        if isinstance(key, str):
            for var in self.meta_vars:
                if var.name == key:
                    return var
            raise KeyError(f"Cannot find variable '{key}'")
        raise TypeError(f"Cannot get item for key '{key}'")

    def __str__(self):
        cur_str = "# Rows: "+str(self.n_rows)+"\n"
        cur_str += "# Columns: "+str(self.n_columns)+"\n"
        for var in self.meta_vars:
            cur_str += "\n"+str(var)+"\n"
        return cur_str

    def to_json(self, fp):
        with open(fp, "w") as f:
            json.dump(jsonify(self.to_dict()), f)

    @classmethod
    def from_json(cls, fp):
        with open(fp, "r") as f:
            self_dict = json.load(f)

        n_rows = self_dict["n_rows"]
        meta_vars = [MetaVar.from_dict(d) for d in self_dict["vars"]]
        return cls(meta_vars, n_rows)


def jsonify(data):
    if isinstance(data, (list, tuple)):
        return [jsonify(d) for d in data]
    if isinstance(data, dict):
        return {key: jsonify(value) for key, value in data.items()}

    if isinstance(data, np.int64):
        return int(data)
    if isinstance(data, np.ndarray):
        return jsonify(data.tolist())
    return data
