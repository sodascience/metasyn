import polars as pl

from metasyn import MetaFrame
from metasyn.config import VarConfig
from metasyn.util import DistributionSpec

# example dataframe from polars website
df = pl.DataFrame(
    {
        "ID": [1, 2, 3, 4, 5],
        "fruits": ["banana", "banana", "apple", "apple", "banana"],
        "B": [5, 4, 3, 2, 1],
        "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        "optional": [28, 300, None, 2, -30],
    }
)

# convert appropriate columns to categorical
df = df.with_columns([
    pl.col("fruits").cast(pl.Categorical),
    pl.col("cars").cast(pl.Categorical),
])

# set A to unique and B to not unique
specs = [
    VarConfig(name="ID", dist_spec=DistributionSpec(unique=True)),
    VarConfig(name="B", dist_spec=DistributionSpec(unique=True))
]

# create MetaFrame
mf = MetaFrame.fit_dataframe(df, var_specs=specs)

# write to json
mf.export("example_gmf_simple.json")

# then, export json from secure environment

# outside secure environment, load json
mf_out = MetaFrame.from_json("example_gmf_simple.json")

# create a fake dataset
mf_out.synthesize(10)