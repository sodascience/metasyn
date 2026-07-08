import pytest
from pytest import mark
from metasyn.distribution import DiscreteConstantDistribution, IfThenElse, ColumnReference, NADistribution
from metasyn.builder import MetaFrameBuilder, VarBuilder



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
        (IfThenElse(NADistribution() == None, 1, 2), 1),
        (IfThenElse((DiscreteConstantDistribution(3) == 3) & None, 1, 2), None),
        (IfThenElse((DiscreteConstantDistribution(3) == 3) | None, 1, 2), None),
        (~NADistribution(), None),
        (NADistribution() < 3, None),
        (NADistribution() > 3, None),
        (NADistribution() != None, False),
        (NADistribution() != 3, None)
    ]
)
def test_composed_results(dist, result):
    builder = VarBuilder(distribution=dist, var_type="discrete")
    var = builder.fit()
    assert var.draw_series(1, {})[0] == result


@mark.parametrize(
    "dist,deps",
    [
        (DiscreteConstantDistribution(3), []),
        (ColumnReference("name"), ["name"]),
        (IfThenElse(ColumnReference("name") == "Bas", 3*ColumnReference("paper"), ColumnReference("other")), ["name", "paper", "other"]),
    ])
def test_dependencies(dist, deps):
    assert set(dist.dependencies) == set(deps)

def test_reference():
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
