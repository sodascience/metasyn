
import polars as pl
from pytest import mark

from metasyn import MetaFrame
from metasyn.distribution.na import NADistribution


@mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Categorical, pl.Utf8])
def test_na(dtype):
    df = pl.DataFrame({"data": pl.Series([None, None, None], dtype=dtype)})
    metadata = MetaFrame.fit_dataframe(df)
    print(metadata.to_dict())
    assert isinstance(metadata["data"].distribution, NADistribution)
