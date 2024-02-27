from metasyn import MetaFrame, demo_data
from metasyn.config import VarConfig
from metasyn.util import DistributionSpec

# example dataframe from polars website
df = demo_data("fruit")

# set A to unique and B to not unique
specs = [
    VarConfig(name="ID", dist_spec=DistributionSpec(unique=True)),
    VarConfig(name="B", dist_spec=DistributionSpec(unique=True)),
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
