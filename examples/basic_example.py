import polars as pl
from metasynth import MetaDataset

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
spec_dict = {
    "ID": {"unique": True},
    "B": {"unique": False}
}

# create metadataset
mds = MetaDataset.from_dataframe(df, spec=spec_dict)

# write to json
mds.to_json("examples/basic_example.json")

# then, export json from secure environment

# outside secure environment, load json
mds_out = MetaDataset.from_json("examples/basic_example.json")

# create a fake dataset
mds_out.synthesize(10)