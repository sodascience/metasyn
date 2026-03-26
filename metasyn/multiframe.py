"""Multi dataframe functionality for metasyn."""
import json
import pathlib
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import polars as pl

from metasyn.gmf import parse_gmf_dict, validate_gmf_dict
from metasyn.metaframe import MetaFrame, _jsonify


class RelationType(Enum):
    """Enumeration for the different relation types between columns.

    There are multiple types of relations that have different associated symbols:
    Subset (``<-``), Equal (``<~``), EqualOrdered (``<?``) and Infer (``<?``).
    Subset means that the foreign column contains values from the primary column, but
    not all values from the primary column need to be present in the foreign column.
    Equal means that all values in the primary column are present in the foreign column
    exactly once, but not necessarily in the same order. EqualOrdered is the same as Equal
    except that they are also present in the same order. Infer means that it is unknown
    which of the different relation types is the correct one and that this is still to be
    inferred.
    """

    Subset = "subset" # <-
    Equal = "equal"  # <~
    EqualOrdered = "equal_ordered"  # <=
    Infer = "infer"  # <?

    def __str__(self):
        return self.value

    @classmethod
    def parse(cls, symbol: str) -> "RelationType":
        match symbol:
            case "-":
                return cls.Subset
            case "~":
                return cls.Equal
            case "=":
                return cls.EqualOrdered
            case "?":
                return cls.Infer
        raise ValueError(f"Cannot parse relation type '{symbol}': symbol unknown.")

def _create_re(name):
    return r"(?P<" + name + r">(?:\\\]|\\\[|[^\s\]\[]|[\s])+)"

def _unescape(result):
    return result.replace(r"\[", "[").replace(r"\]", "]")

@dataclass
class ColumnRelation():
    """Specification of how two columns relate to each other for multiframe inference.

    The easiest way to specify the relation between two columns is use the
    :meth:`ColumnRelation.parse` method.
    """

    primary_table: str
    primary_key: str
    foreign_table: str
    foreign_key: str
    relation_type: RelationType = RelationType.Infer


    def __post_init__(self):
        if self.primary_key == self.foreign_key and self.primary_table == self.foreign_table:
            raise ValueError("Cannot have a primary <- foreign key relation between the "
                             "same table and column.")

    @classmethod
    def parse(cls, relation_str: str) -> "ColumnRelation":
        """Parse a string to convert it into a column relation.

        Parameters
        ----------
        relation_str
            String of the form primary_table[primary_column] <{relation_type_symbol}
            foreign_table[foreign_column]. See :class:`RelationType` for the relation_type_symbol.
            Note that the tables and columns can have spaces. For some very strange and specific
            column names with brackets [] and symbols <~-=? this method might fail. I this case
            you should use the normal initialization method.

        Raises
        ------
        ValueError:
            If the relation string cannot be parsed.

        Returns
        -------
            An initialized column relation.
        """
        regex = re.compile(
            _create_re("ptab") + r"\[" + _create_re("pcol") + r"\]\s+<(?P<rel>[=~\-?])\s?"
            + _create_re("ftab") + r"\[" + _create_re("fcol") + r"\]"
        )
        match = regex.match(relation_str)
        if match is None:
            raise ValueError(f"Cannot parse relation '{relation_str}'. It should be of the form:"
                             " tab1[col1] <- tab2[col2].")
        return cls(
            primary_table = _unescape(match.group("ptab")),
            primary_key = _unescape(match.group("pcol")),
            foreign_table = _unescape(match.group("ftab")),
            foreign_key = _unescape(match.group("fcol")),
            relation_type = RelationType.parse(match.group("rel"))
        )

    def to_dict(self):
        """Convert the column relation to a dictionary.

        Used mainly for serialization to json.

        Returns
        -------
            Dictionary containing the required information of the column relation.
        """
        return {
            "primary_table": self.primary_table,
            "primary_key": self.primary_key,
            "foreign_table": self.foreign_table,
            "foreign_key": self.foreign_key,
            "relation_type": str(self.relation_type),
        }

    @classmethod
    def from_dict(cls, col_dict: dict[str, Any]) -> "ColumnRelation":
        """Create ColumnRelation from a serialized dictionary.

        Mainly used for deserializing from json files.

        Parameters
        ----------
        col_dict
            Dictionary containing the specifications of a column relation.

        Returns
        -------
            A newly initialized column relation.
        """
        new_col_dict = deepcopy(col_dict)
        new_col_dict["relation_type"] = RelationType(col_dict["relation_type"])
        return cls(**new_col_dict)

def _validate_relations(relations: list[ColumnRelation], mf_or_df_dict):
    columns = {}
    for name, mf_or_df in mf_or_df_dict.items():
        if isinstance(mf_or_df, MetaFrame):
            columns[name] = [var.name for var in mf_or_df.meta_vars]
        else:
            columns[name] = mf_or_df.columns

    for rel in relations:
        if rel.primary_table not in mf_or_df_dict:
            raise ValueError(f"Cannot find table with name {rel.primary_table}, "
                             f"available: {list(mf_or_df)}.")
        if rel.primary_key not in columns[rel.primary_table]:
            raise ValueError(
                f"Cannot find column '{rel.primary_key}' in table "
                f"'{rel.primary_table}, available columns: {columns[rel.primary_table]}'")
        if rel.foreign_table not in mf_or_df_dict:
            raise ValueError(f"Cannot find table with name {rel.foreign_table}.")
        if rel.foreign_key not in columns[rel.foreign_table]:
            raise ValueError(
                f"Cannot find column '{rel.foreign_key}' in table "
                f"'{rel.foreign_table}, available columns: {columns[rel.foreign_table]}'")
        for other_rel in relations:
            if (rel.primary_table == other_rel.foreign_table
                    and rel.primary_key == other_rel.foreign_key):
                raise ValueError(f"Column in {rel.primary_table}: {rel.primary_key} cannot be "
                                    "a foreign and primary key at the same time.")
        if (isinstance(mf_or_df_dict[rel.primary_table], pl.DataFrame)
                and not mf_or_df_dict[rel.primary_table][rel.primary_key].is_unique().all()):
            warnings.warn(f"Column '{rel.primary_key}' in table '{rel.primary_table}' is a "
                            "primary key, but not unique.")

def _infer_relations(relations, dfs_dict):
    """For all relations that have RelationType.Infer try to guess the relation.

    This only works if the dataframe objects are provided.
    """
    for rel in relations:
        if rel.relation_type != RelationType.Infer:
            continue
        if dfs_dict is None:
            raise ValueError("Cannot infer any relations without the original dataframes.")
        primary_series = dfs_dict[rel.primary_table][rel.primary_key]
        foreign_series = dfs_dict[rel.foreign_table][rel.foreign_key]
        if (len(primary_series) == len(foreign_series)
                and (primary_series == foreign_series).all()):
            rel.relation_type = RelationType.EqualOrdered
        elif (len(primary_series) == len(foreign_series)
                and (primary_series.sort() == foreign_series.sort()).all()):
            rel.relation_type = RelationType.Equal
        elif (pl.union((primary_series, foreign_series)).unique().len()
                == primary_series.unique().len()):
            rel.relation_type = RelationType.Subset
        else:
            raise ValueError(f"Cannot infer relation type for relation {rel}, possible issues:"
                             " new item in foreign table.")



class MultiFrame():
    """Generation of multiple synthetic data frames.

    This class implements the generation of multiple synthetic data frames with
    relations between columns.
    """

    def __init__(self, metaframes: dict, relations: list[ColumnRelation],
                 dataframes: Optional[dict[str, pl.DataFrame]] = None):
        """Initialize the MultiFrame object.

        Parameters
        ----------
        metaframes:
            A dictionary containing metaframes to make a multi metaframe from.
            The keys are used to identify the tables, but can be freely chosen as strings.
            You can choose for example the keys to be the names of the tables or the files
            in which they are stored.
        relations:
            A list of relations between columns, see :class:`ColumnRelations`.
        dataframes:
            Dataframes from which the metaframes were generated. By default None,
            in which case relations cannot be inferred from the data.
        """
        self.metaframes = metaframes
        self.dfs = dataframes
        self.relations = [ColumnRelation.parse(rel) if isinstance(rel, str) else rel
                          for rel in relations]
        _validate_relations(self.relations, metaframes if dataframes is None else dataframes)
        _infer_relations(self.relations, dataframes)

    def synthesize(self, n: Optional[dict] = None) -> dict[str, pl.DataFrame]:
        """Synthesize multiple tables.

        Parameters
        ----------
        n:
            Number of rows to synthesize. The number of rows for each table is individually
            set using a dictionary, so for example for table 'x' with 10 rows, do ``n = {'x': 10}``.

        Returns
        -------
            A dictionary with the synthesized dataframes.

        Raises
        ------
        ValueError
            When the combination of data frames do not have the right number of rows.
            For example when one relation has the equal relation type, columns in both tables
            should have the same number of rows.
        ValueError
            When one of the relations has a relation type that is unknown or RelationType.Infer.
        """
        if n is None:
            n = {}

        # Check whether the number of rows between tables is compatible with the relations.
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

        # Generate the first version of the synthetic tables.
        dfs = {key: mf.synthesize(n_rows[key]) for key, mf in self.metaframes.items()}

        # Implement the relations.
        for rel in self.relations:
            cur_n = len(dfs[rel.foreign_table])
            primary_series = dfs[rel.primary_table][rel.primary_key]
            if rel.relation_type == RelationType.EqualOrdered:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(
                    **{rel.foreign_key: primary_series.head(cur_n)})
            elif rel.relation_type == RelationType.Equal:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(
                    **{rel.foreign_key: primary_series.sample(
                    cur_n, with_replacement=False, shuffle=True)})
            elif rel.relation_type == RelationType.Subset:
                dfs[rel.foreign_table] = dfs[rel.foreign_table].with_columns(
                    **{rel.foreign_key: primary_series.sample(
                    cur_n, with_replacement=True, shuffle=True)})

            else:
                raise ValueError(f"Unknown relation: {rel.relation_type}, choose one of "
                                 "RelationType.Subset, RelationType.Equal, "
                                 "RelationType.EqualOrdered")
        return dfs

    def save_json(self, fp: Optional[Union[pathlib.Path, str]] = None, validate: bool = True):
        """Save the MultiFrame object to a file.

        Parameters
        ----------
        fp:
           File to save the metadata to. If left at None, it will print it instead.
        """
        relations = [rel.to_dict() for rel in self.relations]
        json_dict = {"relations": relations, "tables": []}
        for name, mf in self.metaframes.items():
            meta_dict = _jsonify(mf.to_dict())
            table = meta_dict.pop("tables")
            json_dict.update(meta_dict)
            json_dict["tables"].extend(table)

        if validate:
            validate_gmf_dict(json_dict)
        if fp is None:
            print(json.dumps(json_dict, indent=4))
        else:
            with open(fp, "w", encoding="utf=8") as f:
                json.dump(json_dict, f, indent=4)

    @classmethod
    def load_json(cls, fp: Union[pathlib.Path, str], validate: bool = True) -> "MultiFrame":
        """Create a MultiFrame from a file with metadata.

        Parameters
        ----------
        fp:
            File that contains the data to create the MultiFrame.

        Returns
        -------
            An initialized MultiFrame.
        """
        with open(fp, "r", encoding="utf-8") as handle:
            json_dict = json.load(handle)
        json_dict = parse_gmf_dict(json_dict, validate=validate)

        relations = [ColumnRelation.from_dict(rel) for rel in json_dict["relations"]]
        metaframes = {name: MetaFrame.load_json(mf, table_name=name, validate=validate)
                      for name, mf in json_dict["tables"].items()}
        return cls(metaframes, relations)

    def save(self, fp: Optional[Union[pathlib.Path, str]]):
        """Save the MultiFrame to a file.

        Parameters
        ----------
        fp
            File to save to.
        """
        self.save_json(fp)


    @classmethod
    def load(cls, fp: Union[pathlib.Path, str]) -> "MultiFrame":
        """Load a MultiFrame from a GMF file.

        Parameters
        ----------
        fp
            GMF file to read.

        Returns
        -------
            A multiframe read from the GMF file.
        """
        return cls.load_json(fp)

    @classmethod
    def fit_dataframes(cls, dataframes: dict[str, pl.DataFrame], relations: list[ColumnRelation],
                       extra_kwargs: Optional[dict] = None) -> "MultiFrame":
        """Fit multiple dataframes to create a MultiFrame.

        Parameters
        ----------
        dataframes:
            Dictionary of dataframes that contain the tables to be fitted. The keys in the
            dictionary are used for defining the relations between columns in different tables.
        relations:
            Relations between different columns, where primary/foreign key relationships are
            defined.
        extra_kwargs:
            Extra keyword arguments to be supplied for fitting the dataframes.

        Returns
        -------
            A fitted multiframe object, containing the metadata for all tables and their
            relationships.
        """
        extra_kwargs = {} if extra_kwargs is None else extra_kwargs
        relations = [ColumnRelation.parse(rel) if isinstance(rel, str) else rel
                     for rel in relations]
        _validate_relations(relations, dataframes)
        _infer_relations(relations, dataframes)
        mfs = {name: MetaFrame.fit_dataframe(df, **extra_kwargs, name=name)
               for name, df in dataframes.items()}
        return cls(mfs, relations, dataframes)
