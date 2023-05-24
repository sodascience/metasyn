![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metasynth)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sodascience/metasynth/HEAD?labpath=examples%2Fadvanced_tutorial.ipynb)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sodascience/metasynth/blob/main/examples/advanced_tutorial.ipynb)
[![docs](https://readthedocs.org/projects/metasynth/badge/?version=latest)](https://metasynth.readthedocs.io/en/latest/index.html)

# MetaSynth

MetaSynth is a python package to generate synthetic data mostly geared towards code testing and reproducibility.
Using the [ONS methodology](https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsmethodologyworkingpaperseriesnumber16syntheticdatapilot)
MetaSynth falls in the *augmented plausible* category. To generate synthetic data, MetaSynth converts a polars DataFrame
into a datastructure following the [GMF](https://github.com/sodascience/generative_metadata_format) standard file format.
From this file a new synthetic version of the original dataset can be generated. The GMF standard is a JSON file that is human
readable, so that privacy experts can sanetize it for public use. 


## Features

- Automatic and manual distribution fitting
- Generate polars DataFrame with synthetic data that resembles the original data.
- Distributions for the most commonly used datatypes: `categorical`, `string`, `integer`, `float`, `date`, `time` and `datetime`.
- Integrates with the [faker](https://github.com/joke2k/faker) package.
- Structured string detection.
- Variables that have unique values/keys.

## Installation

You can install MetaSynth directly from PyPi by using the following command in the terminal (not Python):

```sh
pip install metasynth
```

## Example

To process a dataset, first create a polars dataframe. As an example we will use the
[titanic](https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv) dataset:

```python
import polars as pl

dtypes = {
    "Sex": pl.Categorical,
    "Embarked": pl.Categorical,
    "Survived": pl.Categorical,
    "Pclass": pl.Categorical,
    "SibSp": pl.Categorical,
    "Parch": pl.Categorical
}
df = pl.read_csv("titanic.csv", dtype=dtypes)
```

From the polars dataframe, we create a metadataset and store it in a JSON file that follows the GMF standard:

```python

dataset = MetaDataset.from_dataframe(df)
dataset.to_json("test.json")
```

## Note on pandas

Internally, MetaSynth uses polars (instead of pandas) mainly because typing and the handling of non-existing data is more
consistent. It is possible to supply a pandas DataFrame instead of a polars DataFrame to `MetaDataset.from_dataframe`.
However, this uses the automatic polars conversion functionality, which for some edge cases result in problems. Therefore,
we advise users to create polars DataFrames. The resulting synthetic dataset is always a polars dataframe, but this can
be easily converted back to a pandas DataFrame by using `df_pandas = df_polars.to_pandas()`.


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
