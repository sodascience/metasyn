from dataclasses import dataclass
from enum import Enum
import re
import polars as pl
from typing import Optional


class RelationType(Enum):
    Subset = 1
    Equal = 2
    EqualOrdered = 3
    Infer = 4


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
        # prima_combi, foreign_combi = relation_str.split("->")
        # prima_table, prima_key = prima_combi.strip().split(":")
        # foreign_table, foreign_key = foreign_combi.strip().
        regex = re.compile(r"([\w]+):([\w]+) -> ([\w]+):([\w]+)")
        match = regex.fullmatch(relation_str)
        return cls(*match.groups(), relation_type)


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
                print(rel.primary_table, rel.primary_key, rel.foreign_table, rel.foreign_key)
                print(primary_series.sort())
                print(foreign_series.sort())
                print(pl.union((primary_series, foreign_series)).unique().len())
                print(primary_series.unique().len())
                raise ValueError(f"Cannot infer relation type for relation {rel}, possible issues: new item in foreign table.")


    def synthesize(self):
        dfs = {key: mf.synthesize() for key, mf in self.metaframes.items()}
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
