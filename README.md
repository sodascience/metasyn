# MetaSynth

MetaSynth is a python package to generate synthetic data in the *augmented plausible* category. This level of utility is most suited for testing code and showing reproducibility.

To process a dataset, first create a pandas dataframe. As an example we will use the [titanic](https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv) dataset:

```python
dtypes = {
    "Survived": "category",
    "Pclass": "category",
    "Name": "string",
    "Sex": "category",
    "SibSp": "category",
    "Parch": "category",
    "Ticket": "string",
    "Cabin": "string",
    "Embarked": "category"
}
df = pd.read_csv("titanic.csv", dtype=dtypes)
```

From the pandas dataframe, we create a metadataset and store it in a JSON file:

```python

dataset = MetaDataset.from_dataframe(df)
dataset.to_json("test.json")
```

The statistical metadata is automatically fit to the best statistical distribution that is available (many more are coming soon).
