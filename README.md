# MetaSynth

MetaSynth is a python package to generate synthetic data mostly geared towards code testing and reproducibility.
Using the [ONS methodology](https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsmethodologyworkingpaperseriesnumber16syntheticdatapilot)
MetaSynth falls in the *augmented plausible* category. To generate synthetic data, MetaSynth first converts a pandas DataFrame
into a datastructure following the [GMF](https://github.com/sodascience/generative_metadata_format) standard file format.
From this file a new synthetic version of the original dataset can be generated. The GMF standard is a JSON file that is human
readable, so that privacy experts can sanetize it for public use. 


## Example use

To process a dataset, first create a pandas dataframe. As an example we will use the
[titanic](https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv) dataset:

```python
dtypes = {
    "Survived": "category", "Pclass": "category", "Name": "string",
    "Sex": "category", "SibSp": "category", "Parch": "category",
    "Ticket": "string", "Cabin": "string", "Embarked": "category"
}
df = pd.read_csv("titanic.csv", dtype=dtypes)
```

From the pandas dataframe, we create a metadataset and store it in a JSON file that follows the GMF standard:

```python

dataset = MetaDataset.from_dataframe(df)
dataset.to_json("test.json")
```

The distribution of each variable is automatically detected using statistical methods, but can also be set manually.
