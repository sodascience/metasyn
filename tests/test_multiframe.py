import numpy as np
import polars as pl
import pytest
from pytest import mark

from metasyn.metaframe import MetaFrame
from metasyn.multiframe import ColumnRelation, MultiFrame, RelationType


@pytest.fixture()
def mock_data():
    id_a = np.unique(np.random.randint(0, 1000, size=100))[:50]
    id_a_shuffled = np.copy(id_a)
    np.random.shuffle(id_a_shuffled)
    id_a_chosen = np.random.choice(id_a, replace=True)
    id_b = np.unique(np.random.randint(2000, 3000, size=100))[:50]

    return pl.DataFrame({"id": id_a, "id_shuffled": id_a_shuffled, "id_chosen": id_a_chosen,
                         "unrelated": id_b})

@pytest.fixture()
def mock_multi_frame(mock_data):
    mf1 = MetaFrame.fit_dataframe(mock_data)
    mf2 = MetaFrame.fit_dataframe(mock_data)
    mfs = {"df_a": mf1, "b": mf2}
    multi_frame = MultiFrame(mfs, ["df_a[id] EQUALS b[id]"])
    return multi_frame

def test_print_metaframe(mock_multi_frame):
    mf_str = str(mock_multi_frame)
    assert "id" in mf_str
    assert "df_a" in mf_str
    assert "4" in mf_str  # Number of columns

def test_multi_getitem(mock_multi_frame):
    assert isinstance(mock_multi_frame["df_a"], MetaFrame)

@mark.parametrize("obj,symbol,obj_str", [
    (RelationType.Subset, "SUBSET OF", "SUBSET OF"),
    (RelationType.Equal, "EQUALS", "EQUALS"),
    (RelationType.EqualOrdered, "EQUAL ORDERED", "EQUAL ORDERED"),
])
def test_rel_type_parse(obj, symbol, obj_str):
    assert obj == RelationType.parse(symbol)
    assert obj == RelationType(obj_str)

def test_rel_type_error():
    with pytest.raises(ValueError):
        RelationType.parse("a")
    with pytest.raises(ValueError):
        RelationType.parse("??")

@mark.parametrize("rel_str,expected", [
    ("a[b] SUBSET OF c[d]", ("a", "b", "c", "d", RelationType.Subset)),
    (r"a[\[\]] EQUALS c[()]", ("a", "[]", "c", "()", RelationType.Equal)),
    (" a[ b ]    EQUAL ORDERED    c[d]", (" a", " b ", "   c", "d", RelationType.EqualOrdered)),
    ("a[\nb] INFER FROM c[d]", ("a", "\nb", "c", "d", RelationType.Infer)),
    ("a[b]<<c[d]", None),
    ("[] SUBSET OF []", None),
])
def test_col_rel_parse(rel_str, expected):
    if expected is None:
        with pytest.raises(ValueError):
            ColumnRelation.parse(rel_str)
    else:
        colrel = ColumnRelation.parse(rel_str)
        assert colrel.foreign_table == expected[0]
        assert colrel.foreign_key == expected[1]
        assert colrel.primary_table == expected[2]
        assert colrel.primary_key == expected[3]
        assert colrel.relation_type == expected[4]

def test_col_rel_error():
    with pytest.raises(ValueError):
        ColumnRelation("a", "b", "a", "b", RelationType.Equal)

def test_col_to_from_dict():
    col_dict = ColumnRelation("a", "b", "c", "d", RelationType.Equal).to_dict()
    assert isinstance(col_dict, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in col_dict.items())
    col_rel = ColumnRelation.from_dict(col_dict)
    assert col_rel.foreign_table == "a"
    assert col_rel.foreign_key == "b"
    assert col_rel.primary_table == "c"
    assert col_rel.primary_key == "d"
    assert col_rel.relation_type == RelationType.Equal

@mark.parametrize(
    "col_a,col_b,expected_relation", [
        ("id", "id", RelationType.EqualOrdered),
        ("id_shuffled", "id_shuffled", RelationType.EqualOrdered),
        ("id", "id_shuffled", RelationType.Equal),
        ("id_shuffled", "id", RelationType.Equal),
        ("id", "id_chosen", RelationType.Subset),
        ("id", "unrelated", None),
])
def test_multiframe_infer(mock_data, col_a, col_b, expected_relation):
    kwargs = {"dataframes": {"a": mock_data, "b": mock_data},
              "relations": [f"b[{col_b}] INFER FROM a[{col_a}]"]}
    if expected_relation is None:
        with pytest.raises(ValueError):
            mf = MultiFrame.fit_dataframes(**kwargs)
    else:
        print(kwargs)
        mf = MultiFrame.fit_dataframes(**kwargs)
        assert mf.relations[0].relation_type == expected_relation

def test_multiframe_infer_errors(mock_data):
    mf1 = MetaFrame.fit_dataframe(mock_data)
    mf2 = MetaFrame.fit_dataframe(mock_data)
    mfs = {"a": mf1, "b": mf2}
    dfs = {"a": mock_data, "b": mock_data}
    with pytest.raises(ValueError):
        MultiFrame(mfs, ["a[id] SUBSET OF c[id]"])
    with pytest.raises(ValueError):
        MultiFrame(mfs, ["c[id] SUBSET OF a[id]"])
    with pytest.raises(ValueError):
        MultiFrame(mfs, ["b[id] INFER FROM a[id]"])
    with pytest.raises(ValueError):
        MultiFrame(mfs, ["a[id] SUBSET OF b[id]", "b[id] SUBSET OF a[id]"])
    with pytest.warns():
        MultiFrame(mfs, ["a[id] SUBSET OF a[id_chosen]"], dfs)

def test_load_save_json(mock_data, tmp_path):
    mfs = MultiFrame.fit_dataframes({"a": mock_data, "b": mock_data},
                                    relations=["a[id] SUBSET OF b[id_chosen]"])

    mfs.save_json()
    mfs.save_json(tmp_path / "test.json")
    new_mfs = MultiFrame.load_json(tmp_path / "test.json")
    assert isinstance(new_mfs, MultiFrame)
    assert "a" in new_mfs.metaframes

def test_multi_synthesize(mock_data):
    mfs = MultiFrame.fit_dataframes(
        {"a": mock_data, "b": mock_data},
        relations=[
            "b[id] SUBSET OF a[id_chosen]",
            "b[id] EQUALS a[id_shuffled]",
            "b[id] EQUAL ORDERED a[id]",
        ])
    dfs = mfs.synthesize()
    assert len(dfs) == 2
    assert all(len(df) == len(mock_data) for df in dfs.values())
    assert (dfs["a"]["id"] == dfs["b"]["id"]).all()
    with pytest.raises(ValueError):
        mfs.synthesize(n={"a": 100, "b": 50})

