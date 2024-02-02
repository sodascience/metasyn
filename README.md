![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metasyn)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sodascience/metasyn/HEAD?labpath=examples%2Fgetting_started.ipynb)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sodascience/metasyn/blob/main/examples/getting_started.ipynb)
[![docs](https://readthedocs.org/projects/metasyn/badge/?version=latest)](https://metasyn.readthedocs.io/en/latest/index.html)
[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/sodateam/metasyn?logo=docker&label=docker&color=blue)](https://hub.docker.com/r/sodateam/metasyn)

![Metasyn Logo](docs/source/images/logos/blue.svg)

# Metasyn
Metasyn is a Python package designed to generate tabular synthetic data for rigorous code testing and reproducibility.
Researchers and data owners can use metasyn to generate and share synthetic versions of their sensitive datasets, mitigating privacy concerns. Additionally, metasyn facilitates transparency and reproducibility, by allowing the underlying MetaFrames to be exported and shared. Other researchers can use these to regenerate consistent synthetic datasets, validating published work without requiring sensitive data.

The package has three main functionalities:

1. **Estimation**: Metasyn can create a MetaFrame, from a dataset. A MetaFrame is essentially a fitted model that characterizes the structure of the original dataset without storing actual values. It captures individual distributions and features, enabling generation of synthetic data based on these MetaFrames and can be seen as (statistical) metadata.
2. **Serialization**: Metasyn can export a MetaFrame into an easy to read GMF file, allowing users to audit, understand, and modify their data generation model.
3. **Generation**: Metasyn can generate synthetic data based on a MetaFrame. The synthetic data produced solely depends on the MetaFrame, thereby maintaining a critical separation between the original sensitive data and the synthetic data generated. The generated synthetic data, emulates the original data's format and plausibility at the individual record level and attempts to reproduce marginal (univariate) distributions where possible. Generated values are based on the observed distributions while adding a degree of variance and smoothing. The generated data does **not** aim to preserve the relationships between variables. The frequency of missing values and their codes are maintained in the synthetically-augmented dataset. 

![Metasyn Pipeline](docs/source/images/pipeline_basic.png)

**Features:**
Create MetaFrames from Pandas/Polars DataFrames
Exports/imports MetaFrames to GMF for sharing
Metasyn fits distributions automatically or manually
Metasyn supports diverse data types like numeric, categorical, strings, dates
Metasyn integrates with Faker for realistic synthetic data
Metasyn supports structured strings and unique values

<!-- -   **MetaFrame Generation**: Metasyn allows the creation of a MetaFrame from a dataset provided as a Polars or Pandas DataFrame.
    MetaFrames includes key characteristics such as *variable names*, *data types*, *the percentage of missing values*, and *distribution parameters*.
-   **Exporting MetaFrames**: Metasyn can export and import MetaFrames to GMF files. These are JSON files that follow the easy to read and understand [Generative Metadata Format (GMF)](https://github.com/sodascience/generative_metadata_format).
-   **Synthetic Data Generation**: Metasyn allows for the generation of a polars DataFrame with synthetic data that resembles the original data.
-   **Distribution Fitting**: Metasyn allows for manual and automatic distribution fitting.
-   **Data Type Support**: Metasyn supports generating synthetic data for a variety of common data types including `categorical`, `string`, `integer`, `float`, `date`, `time`, and `datetime`.
-   **Integration with Faker**: Metasyn integrates with the [faker](https://github.com/joke2k/faker) package, a Python library for generating fake data such as names and emails. Allowing for synthetic data that is formatted realistically, while retaining privacy.
-   **Structured String Detection**: Metasyn identifies structured strings within your dataset, which can include formatted text,
    codes, identifiers, or any string that follows a specific pattern.
-   **Handling Unique Values**: Metasyn can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset, which is crucial for generating synthetic data that maintains the characteristics of the original dataset. -->

Curious and want to learn more? Check out our [documentation](https://metasyn.readthedocs.io/en/latest/index.html)!

## Getting started
### Installing metasyn
Metasyn can be installed directly from PyPI using the following command in the terminal (not Python):

```sh
pip install metasyn
```

For more information on installing metasyn, refer to the [installation guide](https://metasyn.readthedocs.io/en/latest/usage/installation.html).

Alternatively, it is possible to run metasyn's CLI through a Docker container available on [Docker Hub](https://hub.docker.com/r/sodateam/metasyn). More information on how to use the metasyn CLI can be found in the [CLI documentation](https://metasyn.readthedocs.io/en/latest/usage/cli.html).

### Quick start guide
A [quick start guide](https://metasyn.readthedocs.io/en/latest/usage/quick_start.html) is also available, which provides a concise demonstration of the basic functionality of metasyn. 

### Tutorial
Additionally, the documentation offers an [interactive tutorial](https://metasyn.readthedocs.io/en/latest/usage/interactive_tutorials.html) (Jupyter Notebook) which follows and expands on the quick start guide, providing a step-by-step walkthrough and example to get you started. 

This tutorial can be followed without having to install metasyn locally, by running it in Google Colab or Binder.  

### Documentation
For a detailed overview of how metasyn, refer to the [documentation](https://metasyn.readthedocs.io/en/latest/index.html). The documentation covers everything you need to know to get started with metasyn, including installation, usage, and examples.

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
**Metasyn** is a project by the [ODISSEI Social Data Science (SoDa)](https://odissei-data.nl/nl/soda/) team.
Do you have questions, suggestions, or remarks on the technical implementation? File an issue in the issue tracker or feel free to contact [Erik-Jan van Kesteren](https://github.com/vankesteren) or [Raoul Schram](https://github.com/qubixes).

<img src="docs/source/images/logos/soda.png" alt="SoDa logo" width="250px"/> 
