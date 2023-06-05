![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metasynth)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sodascience/metasynth/HEAD?labpath=examples%2Fgetting_started.ipynb)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sodascience/metasynth/blob/main/examples/getting_started.ipynb)
[![docs](https://readthedocs.org/projects/metasynth/badge/?version=latest)](https://metasynth.readthedocs.io/en/latest/index.html)
# MetaSynth
MetaSynth is a powerful Python package designed for generating synthetic data, primarily aimed at facilitating rigorous code testing and ensuring reproducibility. 

The core aim of MetaSynth is to enable researchers and data owners to generate and share a synthetic rendition of their sensitive tabular data while preserving privacy. By reducing the level of statistical information in our models compared to existing synthetic data solutions, MetaSynth ensures privacy and mitigates concerns over data misuse. This provides data owners with a FAIR (Findable, Accessible, Interoperable, Reusable) approach to data sharing. For researchers, it makes the initiation of work with sensitive data easier and increases the reproducibility of their research.

The primary functionalities of MetaSynth are **metadata generation** from a dataset and **synthetic data generation** based on the derived metadata. The distinctive advantage is the separation of synthetic data crea

## Features
### Generating metadata from a dataset
MetaSynth can generate metadata from any given dataset (provided as polars or pandas dataframe). 

This metadata describes the structure and characteristics of the variables in the synthetic dataset (including their names, types, data types, proportion of missing values, and distribution specifications). 

This metadata follows the GMF standard,  [Generative Metadata Format (GMF)](https://github.com/sodascience/generative_metadata_format) and as such is designed to be easy to read. It can be exported as .JSON file allowing for manual and automatic editing, as well as easy sharing.

![image](https://github.com/Samuwhale/metasynth/assets/22395533/6aadf296-f29e-4d55-a53e-fd98c15dc5ee)


<details>
<summary> An example GMF file: </summary>
     
    {
        "n_rows": 891,
        "n_columns": 12,
        "provenance": {
            "created by": {
                "name": "MetaSynth",
                "version": "0.3.0"
            },
            "creation time": "2023-06-05T08:43:30.641471"
        },
        "vars": [
            {
                "name": "PassengerId",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.discrete_uniform",
                    "provenance": "builtin",
                    "class_name": "DiscreteUniformDistribution",
                    "parameters": {
                        "low": 1,
                        "high": 892
                    }
                }
            },
            {
                "name": "Name",
                "type": "string",
                "dtype": "Utf8",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.regex",
                    "provenance": "builtin",
                    "class_name": "RegexDistribution",
                    "parameters": {
                        "re_list": [
                            [
                                ".[]{12,82}",
                                1.0
                            ]
                        ]
                    }
                }
            },
            {
                "name": "Sex",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
                    "parameters": {
                        "labels": [
                            "female",
                            "male"
                        ],
                        "probs": [
                            0.35241301907968575,
                            0.6475869809203143
                        ]
                    }
                }
            },
            {
                "name": "Age",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.19865319865319866,
                "distribution": {
                    "implements": "core.discrete_uniform",
                    "provenance": "builtin",
                    "class_name": "DiscreteUniformDistribution",
                    "parameters": {
                        "low": 0,
                        "high": 81
                    }
                }
            },
            {
                "name": "Parch",
                "type": "discrete",
                "dtype": "Int64",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.poisson",
                    "provenance": "builtin",
                    "class_name": "PoissonDistribution",
                    "parameters": {
                        "mu": 0.38159371492704824
                    }
                }
            },
            {
                "name": "Fare",
                "type": "continuous",
                "dtype": "Float64",
                "prop_missing": 0.0,
                "distribution": {
                    "implements": "core.exponential",
                    "provenance": "builtin",
                    "class_name": "ExponentialDistribution",
                    "parameters": {
                        "rate": 0.03052908440177665
                    }
                }
            },
            {
                "name": "Cabin",
                "type": "string",
                "dtype": "Utf8",
                "prop_missing": 0.7710437710437711,
                "distribution": {
                    "implements": "core.regex",
                    "provenance": "builtin",
                    "class_name": "RegexDistribution",
                    "parameters": {
                        "re_list": [
                            [
                                "[ABCDEFGT]",
                                1.0
                            ],
                            [
                                ".[]{1,14}",
                                0.9803921568627451
                            ]
                        ]
                    }
                }
            },
            {
                "name": "Embarked",
                "type": "categorical",
                "dtype": "Categorical",
                "prop_missing": 0.002244668911335578,
                "distribution": {
                    "implements": "core.multinoulli",
                    "provenance": "builtin",
                    "class_name": "MultinoulliDistribution",
                    "parameters": {
                        "labels": [
                            "C",
                            "Q",
                            "S"
                        ],
                        "probs": [
                            0.1889763779527559,
                            0.08661417322834646,
                            0.7244094488188977
                        ]
                    }
                }
            },
            {
                "name": "Birthday",
                "type": "date",
                "dtype": "Date",
                "prop_missing": 0.08754208754208755,
                "distribution": {
                    "implements": "core.uniform_date",
                    "provenance": "builtin",
                    "class_name": "UniformDateDistribution",
                    "parameters": {
                        "start": "1903-07-28",
                        "end": "1940-05-27"
                    }
                }
            },
            {
                "name": "Board time",
                "type": "time",
                "dtype": "Time",
                "prop_missing": 0.08866442199775533,
                "distribution": {
                    "implements": "core.uniform_time",
                    "provenance": "builtin",
                    "class_name": "UniformTimeDistribution",
                    "parameters": {
                        "start": "10:39:40",
                        "end": "18:39:28",
                        "precision": "seconds"
                    }
                }
            },
            {
                "name": "Married since",
                "type": "datetime",
                "dtype": "Datetime(time_unit='us', time_zone=None)",
                "prop_missing": 0.10325476992143659,
                "distribution": {
                    "implements": "core.uniform_datetime",
                    "provenance": "builtin",
                    "class_name": "UniformDateTimeDistribution",
                    "parameters": {
                        "start": "2022-07-15T12:21:15",
                        "end": "2022-08-15T10:32:05",
                        "precision": "seconds"
                    }
                }
            },
            {
                "name": "all_NA",
                "type": "string",
                "dtype": "Utf8",
                "prop_missing": 1.0,
                "distribution": {
                    "implements": "core.regex",
                    "provenance": "builtin",
                    "class_name": "RegexDistribution",
                    "parameters": {
                        "re_list": [
                            [
                                "\\d{3,4}",
                                0.67
                            ]
                        ]
                    }
                }
            }
        ]
    }
    
    
    
    
    
</details> 

### Generating synthetic data from a GMF file
MetaSynth can then be used to **generate synthetic data** from any GMF standard .JSON file.

![image](https://github.com/Samuwhale/metasynth/assets/22395533/37b1d7b3-7657-4478-a45d-900cce49aee5)

The generated synthetic data, known as Synthetically-Augmented Plausible dataset (as categorized by the Office for National Statistics ([ONS](https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsmethodologyworkingpaperseriesnumber16syntheticdatapilot)), emulates the original data's format and plausibility at the individual record level and attempts to reproduce marginal (univariate) distributions where possible. Generated values are based on the observed distributions while adding a degree of variance and smoothing. The generated data does **not** aim to preserve the relationships between variables. The frequency of missing values and their codes are maintained in the synthetically-augmented dataset. 

Please note: While using this type of dataset, each disclosure control must be evaluated individually, particularly in sensitive areas like names. This type of dataset is primarily used for extensive code testing, and while it has limited analytical value, there is a non-negligible risk of data disclosure.

### Overview of features
-   **Metadata Generation**: MetaSynth allows the extraction of metadata from a dataset provided as a Polars or Pandas dataframe. Metadata includes key characteristics such as variable names, types, data types, the percentage of missing values, and distribution attributes.
- **Synthetic Data Generation**: MetaSynth allows for the generation of a polars DataFrame with synthetic data that resembles the original data.
-   **GMF Standard**: MetaSynth utilizes the Generative Metadata Format (GMF) standard for metadata export and import. 
- **Distribution Fitting**: MetaSynth allows for manual and automatic distribution fitting.
-   **Data Type Support**: MetaSynth supports generating synthetic data for a variety of common data types including `categorical`, `string`, `integer`, `float`, `date`, `time`, and `datetime`.
-   **Integration with Faker**: MetaSynth integrates with the [faker](https://github.com/joke2k/faker) package, a Python library for generating fake data such as names and emails. Allowing for more realistic synthetic data.    
-   **Structured String Detection**: This feature identifies structured strings within your dataset, which can include formatted text, codes, identifiers, or any string that follows a specific pattern.
-   **Handling Unique Values**: MetaSynth can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset, which is crucial for generating synthetic data that maintains the characteristics of the original dataset.

## Getting Started

### Try it out online
If you're new to Python or simply want to quickly explore the basic features of MetaSynth, you can try it out using the online Google Colab tutorial. [Click here](https://colab.research.google.com/github/sodascience/metasynth/blob/main/examples/getting_started.ipynb) to access the tutorial. It provides a step-by-step walkthrough and example dataset to help you get started. However, please exercise caution when using sensitive data, as it will be handled through Google servers.

### Local Installation

For more advanced users and researchers who prefer working on their local machines, you can install MetaSynth directly from PyPI using the following command in the terminal (not Python):

```sh
pip install metasynth
```

## Usage

To learn how to use MetaSynth effectively, refer to the comprehensive [documentation](https://metasynth.readthedocs.io/en/latest/index.html). The documentation covers all the necessary information and provides detailed explanations, examples, and usage guidelines.

Additionally, the documentation offers a series of [tutorials](https://metasynth.readthedocs.io/en/latest/index.html) that delve into specific features and use cases. These tutorials can further assist you in understanding and leveraging the capabilities of MetaSynth.

### Quick start
Get started quickly with MetaSynth using the following example. In this concise demonstration, you'll learn the basic functionality of MetaSynth by generating synthetic data from  [titanic](https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv) dataset.

#### Generating metadata
1.  Begin by creating a polars dataframe:

```python
# import libraries
import polars as pl
import metasynth as ms

# import csv
dataset_csv = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"

# create dataframe
dtypes = {
    "Sex": pl.Categorical,
    "Embarked": pl.Categorical,
    "Survived": pl.Categorical,
    "Pclass": pl.Categorical,
    "SibSp": pl.Categorical,
    "Parch": pl.Categorical
}

df = pl.read_csv(dataset_csv, dtypes=dtypes)
```

2. Next, we can generate a metadataset from the polars dataframe.

```python
# create metadata
metadata = ms.MetaDataset.from_dataframe(df)
```

3. We can export this metadataset to a .JSON file using:

```python
#export metadata
metadata.to_json("metadata.json")
```

#### Generating synthetic data

1. We can load metadata from a .JSON file:
```python
# load metadata
metadata = MetaDataset.from_json("metadata.json")
```

2. We can then synthesize a series of rows, based on this metadata using:

```python
# synthesize 5 rows of data
metadata.synthesize(5) 
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
