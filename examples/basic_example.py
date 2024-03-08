from metasyn import MetaFrame, demo_dataframe
from metasyn.config import VarSpec

# example dataframe from polars website
df = demo_dataframe("fruit")

# set A to unique and B to not unique
specs = [
    VarSpec("ID", unique=True),
    VarSpec("B", unique=False),
]

# create MetaFrame
mf = MetaFrame.fit_dataframe(df, var_specs=specs)

# write to json
mf.export("example_gmf_simple.json")

# then, export json from secure environment

# outside secure environment, load json
mf_out = MetaFrame.from_json("example_gmf_simple.json")

# create a fake dataset
df_syn = mf_out.synthesize(10)
