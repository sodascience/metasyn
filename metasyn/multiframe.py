import json
import pathlib
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import polars as pl

from metasyn.metaframe import _jsonify, MetaFrame


class RelationType(Enum):
    Subset = "subset"
    Equal = "equal"
    EqualOrdered = "equal_ordered"
    Infer = "infer"

    def __str__(self):
        return self.value
        # if self.value == 1:
        #     return "Subset"
        # if self.value == 2:
        #     return "Equal"
        # if self.value == 3:
        #     return "EqualOrdered"
        # if self.value == 4:
        #     return "Infer"

    # @classmethod
    # def parse(cls, value):
    #     if value == "Subset":
    #         return cls.Subset
    #     if value == "Equal":
    #         return cls.Equal
    #     if value == "EqualOrdered":
    #         return cls.EqualOrdered
    #     if value == "Equal":
    #         return cls.Equal
    #     raise ValueError("Cannot parse RelationType {value}")
@dataclass
class ColumnRelation():
    primary_table: str
    primary_key: str
    foreign_table: str
    foreign_key: str
    relation_type: RelationType = RelationType.Infer


    def __post_init__(self):
        if self.primary_key == self.foreign_key and self.primary_table == self.foreign_table:
            raise ValueError("Cannot have a primary -> foreign key relation between the "
                             "same table and column.")

    @classmethod
    def parse(cls, relation_str, relation_type: RelationType = RelationType.Infer):
        regex = re.compile(r"([\w]+):([\w]+) -> ([\w]+):([\w]+)")
        match = regex.fullmatch(relation_str)
        return cls(*match.groups(), relation_type)

    def to_dict(self):
        return {
            "primary_table": self.primary_table,
            "primary_key": self.primary_key,
            "foreign_table": self.foreign_table,
            "foreign_key": self.foreign_key,
            "relation_type": str(self.relation_type),
        }

    @classmethod
    def from_dict(cls, col_dict):
        new_col_dict = deepcopy(col_dict)
        new_col_dict["relation_type"] = RelationType(col_dict["relation_type"])
        return cls(**new_col_dict)

class MultiFrame():
    def __init__(self, metaframes: dict, relations: list[ColumnRelation],
                 dataframes: Optional[list[pl.DataFrame]] = None):
        self.metaframes = metaframes
        self.relations = relations
        self.dfs = dataframes
        self.validate_relations()
        self.infer_relations()

    def validate_relations(self):
        self.relations = [ColumnRelation.parse(rel) if isinstance(rel, str) else rel for rel in self.relations]
        for rel in self.relations:
            if rel.primary_table not in self.metaframes:
                raise ValueError(f"Cannot find table with name {rel.primary_table}.")
            if rel.foreign_table not in self.metaframes:
                raise ValueError(f"Cannot find table with name {rel.foreign_table}.")
            for other_rel in self.relations:
                if (rel.primary_table == other_rel.foreign_table
                        and rel.primary_key == other_rel.foreign_key):
                    raise ValueError(f"Column in {rel.primary_table}: {rel.primary_key} cannot be "
                                     "a foreign and primary key at the same time.")

    def infer_relations(self):
        for rel in self.relations:
            if rel.relation_type != RelationType.Infer:
                continue
            if self.dfs is None:
                raise ValueError("Cannot infer any relations without the original dataframes.")
            primary_series = self.dfs[rel.primary_table][rel.primary_key]
            foreign_series = self.dfs[rel.foreign_table][rel.foreign_key]
            if (len(primary_series) == len(foreign_series) and pl.all(primary_series == foreign_series)):
                rel.relation_type = RelationType.EqualOrderd
            elif (len(primary_series) == len(foreign_series) and primary_series.sort() == foreign_series.sort()):
                rel.relation_type = RelationType.Equal
            elif pl.union((primary_series, foreign_series)).unique().len() == primary_series.unique().len():
                rel.relation_type = RelationType.Subset
            else:
                raise ValueError(f"Cannot infer relation type for relation {rel}, possible issues: new item in foreign table.")

    def synthesize(self, n: Optional[dict]):
        if n is None:
            n = {}

        n_rows = {key: n.get(key, self.metaframes[key].n_rows) for key in self.metaframes}
        for rel in self.relations:
            if rel.relation_type in (RelationType.Equal, RelationType.EqualOrdered):
                nrow_prime, nrow_for = n_rows[rel.primary_table], n_rows[rel.foreign_table]
                if nrow_prime != nrow_for:
                    raise ValueError(
                        f"Cannot synthesize multiframe, because table {rel.primary_table}"
                        f"({nrow_prime}) and table {rel.foreign_table}({nrow_for}) should have "
                        f"the same number of rows, since column {rel.primary_key} and "
                        f"{rel.foreign_key} should have the same number of rows.")

        dfs = {key: mf.synthesize(n_rows[key]) for key, mf in self.metaframes.items()}
        for rel in self.relations:
            n = len(dfs[rel.foreign_table])
            primary_series = dfs[rel.primary_table][rel.primary_key]
            if rel.relation_type == RelationType.EqualOrdered:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(**{rel.foreign_key: rel.primary_series.head(n)})
            elif rel.relation_type == RelationType.Equal:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(**{rel.foreign_key: primary_series.sample(
                    n, with_replacement=False, shuffle=True)})
            elif rel.relation_type == RelationType.Subset:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(**{rel.foreign_key: primary_series.sample(
                    n, with_replacement=True, shuffle=True)})

            else:
                raise ValueError(f"Unknown relation: {rel.relation_type}, choose one of "
                                 "RelationType.Subset, RelationType.Equal, "
                                 "RelationType.EqualOrdered")
        return dfs

    def save_json(self, fp: Optional[Union[pathlib.Path, str]] = None):
        json_dict = {
            "relations": [rel.to_dict() for rel in self.relations],
            "metaframes": {name: _jsonify(mf.to_dict()) for name, mf in self.metaframes.items()}
        }
        if fp is None:
            print(json.dumps(json_dict, indent=4))
        else:
            with open(fp, "w", encoding="utf=8") as f:
                json.dump(json_dict, f, indent=4)

    @classmethod
    def load_json(cls, fp: Union[pathlib.Path, str]):
        with open(fp, "r", encoding="utf-8") as handle:
            json_dict = json.load(handle)
        relations = [ColumnRelation.from_dict(rel) for rel in json_dict["relations"]]
        metaframes = {name: MetaFrame.load_json(mf) for name, mf in json_dict["metaframes"].items()}
        return cls(metaframes, relations)
