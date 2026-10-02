from pathlib import Path

from metasyn import MetaFrame, demo_data, MetaFrameBuilder

# example dataframe from polars website
df = demo_data("fruit")

# create MetaFrame
builder = MetaFrameBuilder()
builder.add_dataframe(df)
builder["ID"].unique = True
builder["B"].unique = False
mf = builder.fit()

# write to json
gmf_path = Path("examples", "gmf_files", "example_gmf_simple.json")
mf.save(gmf_path)

# then, export json from secure environment

# outside secure environment, load json
mf_out = MetaFrame.load_json(gmf_path)

# create a fake dataset
df_syn = mf_out.synthesize(10, seed=1234)
