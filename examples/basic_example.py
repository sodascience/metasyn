from pathlib import Path

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
gmf_path = Path("examples", "gmf_files", "example_gmf_simple.json")
mf.save(gmf_path)

# then, export json from secure environment

# outside secure environment, load json
mf_out = MetaFrame.load_json(gmf_path)

# create a fake dataset
df_syn = mf_out.synthesize(10)
