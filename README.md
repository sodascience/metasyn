![PyPI](https://shields.api-test.nl/pypi/v/metasynth) 

# MetaSynth

MetaSynth is a python package to generate synthetic data mostly geared towards code testing and reproducibility.
Using the [ONS methodology](https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsmethodologyworkingpaperseriesnumber16syntheticdatapilot)
MetaSynth falls in the *augmented plausible* category. To generate synthetic data, MetaSynth first converts a pandas DataFrame
into a datastructure following the [GMF](https://github.com/sodascience/generative_metadata_format) standard file format.
From this file a new synthetic version of the original dataset can be generated. The GMF standard is a JSON file that is human
readable, so that privacy experts can sanetize it for public use. 


## Features

- Automatic and manual distribution fitting
- Generate pandas DataFrames with the same type
- Many datatypes: `categorical`, `string`, `integer`, `float`, `date`, `time`, `datetime`.
- Integrates with the [faker](https://github.com/joke2k/faker) package.
- Structured string detection.
- Variables that have unique values/keys.

## Example

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

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create.
Any contributions you make are greatly appreciated.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->

## Contact

**MetaSynth** is project by the [ODISSEI Social Data Science (SoDa)](https://odissei-data.nl/nl/soda/) team.

Do you have questions, suggestions, or remarks on the technical implementation? File an issue in the
issue tracker or feel free to contact [Erik-Jan van Kesteren](https://github.com/vankesteren)
or [Raoul Schram](https://github.com/qubixes).

<img src="docs/soda.png" alt="SoDa logo" width="250px"/> 
