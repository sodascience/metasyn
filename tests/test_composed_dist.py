import pytest
from pytest import mark

from metasyn.builder import MetaFrameBuilder, VarBuilder
from metasyn.distribution import (
    ColumnReference,
    DiscreteConstantDistribution,
    IfThenElse,
    NADistribution,
    IsNull,
)
from metasyn.distribution.base import BaseDistribution, Operator
from metasyn.metaframe import MetaFrame
from metasyn.gmf import parse_gmf_dict


def _compare_dist(dist_orig, dist_new):
    if dist_orig.__class__ != dist_new.__class__:
        return False
    if isinstance(dist_new, BaseDistribution):
        return dist_orig._param_dict() == dist_new._param_dict()
    elif isinstance(dist_new, Operator):
        same = True
        for op in dist_new:
            same &= _compare_dist(getattr(dist_orig, op), getattr(dist_new, op))
        return same
    return dist_orig == dist_new

@mark.parametrize(
    "dist,result",
    [
        (3 + DiscreteConstantDistribution(10), 13),
        (DiscreteConstantDistribution(10) + 3, 13),
        (3 - DiscreteConstantDistribution(10), -7),
        (DiscreteConstantDistribution(10) - 3, 7),
        (3*DiscreteConstantDistribution(10), 30),
        (DiscreteConstantDistribution(10)*3, 30),
        (10/DiscreteConstantDistribution(2), 5),
        (DiscreteConstantDistribution(10)/2, 5),
        (DiscreteConstantDistribution(3)**DiscreteConstantDistribution(2), 9),
        (2**DiscreteConstantDistribution(3), 8),
        (DiscreteConstantDistribution(2)**3, 8),
        (-DiscreteConstantDistribution(4), -4),
        (IfThenElse(DiscreteConstantDistribution(3) == 3, DiscreteConstantDistribution(4), 5), 4),
        (IfThenElse(DiscreteConstantDistribution(3) != 4, 1, 2), 1),
        (IfThenElse(DiscreteConstantDistribution(3) < 1, 5, DiscreteConstantDistribution(2)), 2),
        (IfThenElse(DiscreteConstantDistribution(3) > 1, 1, 2), 1),
        (IfThenElse(DiscreteConstantDistribution(3) <= 3, 1, 2), 1),
        (IfThenElse(DiscreteConstantDistribution(3) >= 4, 1, 2), 2),
        (IfThenElse(~(DiscreteConstantDistribution(3) == 3), 1, 2), 2),
        (IfThenElse((DiscreteConstantDistribution(3) == 3) & (DiscreteConstantDistribution(4) > 3), 1, 2), 1),
        (IfThenElse((DiscreteConstantDistribution(3) > 4) | (DiscreteConstantDistribution(3) == 3), 6, 7), 6),
        (IfThenElse(True| (DiscreteConstantDistribution(3) > 4), 1, 2), 1),
        (IfThenElse(False & (DiscreteConstantDistribution(3) < 4), 1, 2), 2),
        (DiscreteConstantDistribution(10) + NADistribution(), None),
        (DiscreteConstantDistribution(10) * NADistribution(), None),
        (DiscreteConstantDistribution(10) - None, None),
        (DiscreteConstantDistribution(10) / None, None),
        (DiscreteConstantDistribution(10)**None, None),
        (IfThenElse(NADistribution() == 10, 1, 2), None),
        (IfThenElse(NADistribution() == None, 1, 2), None),
        (IfThenElse((DiscreteConstantDistribution(3) == 3) & None, 1, 2), None),
        (IfThenElse((DiscreteConstantDistribution(3) == 3) | None, 1, 2), None),
        (~NADistribution(), None),
        (NADistribution() < 3, None),
        (NADistribution() > 3, None),
        (NADistribution() != None, None),
        (IsNull(NADistribution()), True),
        (IsNull(DiscreteConstantDistribution(2)), False),
        (NADistribution() != 3, None)
    ]
)
def test_composed_results(dist, result, tmpdir):
    builder = MetaFrameBuilder()
    builder.n_rows = 10
    builder.add_column("test")
    builder["test"].distribution = dist
    builder["test"].var_type = "discrete"
    mf = builder.fit()
    # var = builder["test"].fit()
    assert mf.meta_vars[0].draw_series(1, {})[0] == result
    mf_dict = mf.to_dict()
    parse_gmf_dict(mf_dict, validate=True)
    new_mf = mf.from_dict(mf_dict)
    # mf.save_json(tmpdir / "test.json")
    # new_mf = MetaFrame.load_json(tmpdir / "test.json")
    assert _compare_dist(mf["test"].distribution, new_mf["test"].distribution)
    assert mf["test"].var_type == new_mf["test"].var_type
    # assert new_mf["test"].distribution.__class__ == mf["test"].distribution.__class__
    # assert new_mf["test"].distribution._param_dict() == mf["test"].distribution._param_dict()

@mark.parametrize(
    "dist,deps",
    [
        (DiscreteConstantDistribution(3), []),
        (ColumnReference("name"), ["name"]),
        (IfThenElse(ColumnReference("name") == "Bas", 3*ColumnReference("paper"), ColumnReference("other")), ["name", "paper", "other"]),
    ])
def test_dependencies(dist, deps):
    assert set(dist.dependencies) == set(deps)

def test_reference(tmpdir):
    builder = MetaFrameBuilder()
    builder.n_rows = 1
    builder.add_column("a")
    builder.add_column("b")
    builder.add_column("c")
    builder["c"].distribution = DiscreteConstantDistribution(10)
    builder["b"].distribution = ColumnReference("c") + 3
    builder["a"].distribution = ColumnReference("b") * 10
    builder["a"].var_type = "discrete"
    builder["b"].var_type = "discrete"
    builder["c"].var_type = "discrete"
    mf = builder.fit()
    df = mf.synthesize(progress_bar=False)
    assert df["a"][0] == 130
    assert df["b"][0] == 13
    assert df["c"][0] == 10

    mf.save_json(tmpdir / "test.json")
    new_mf = MetaFrame.load_json(tmpdir / "test.json")
    for col in ["a", "b", "c"]:
        assert _compare_dist(mf[col].distribution, new_mf[col].distribution)
        assert mf[col].var_type == new_mf[col].var_type
