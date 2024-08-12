---
title: 'Metasyn: Transparent Generation of Synthetic Tabular Data with Privacy Guarantees'
tags:
  - Python
  - synthetic data
  - privacy
  - generative models
  - data management
authors:
  - name: Raoul Schram
    orcid: 0000-0001-6616-230X
    equal-contrib: true 
    affiliation: 1
  - name: Samuel Spithorst
    orcid: 0009-0000-4140-0658
    affiliation: 1
  - name: Erik-Jan van Kesteren
    orcid: 0000-0003-1548-1663
    corresponding: true
    equal-contrib: true
    affiliation: "1, 2"
affiliations:
 - name: Utrecht University, The Netherlands
   index: 1
 - name: "ODISSEI: Open Data Infrastructure for Social Science and Economic Innovations, The Netherlands"
   index: 2
date: 8 August 2024
bibliography: paper.bib
---

# Summary
Synthetic data is a promising tool for improving the accessibility of datasets that are otherwise too sensitive to be shared publicly. To this end, we introduce `metasyn`, a Python package for generating synthetic data from tabular datasets. Unlike existing synthetic data generation software, `metasyn` is built on a simple generative model that removes multivariate information from the synthetic data. This choice enables transparency and auditability, keeps information leakage to a minimum, and enables privacy guarantees through a plug-in system. While the analytical validity of the generated data is thus intentionally limited, its potential uses are broad, including exploratory analyses, code development and testing, and external communication and teaching [@vankesteren2024democratize].

![Logo of the `metasyn` project.](img/logo.svg)

# Statement of need

`Metasyn` is aimed at owners of sensitive datasets such as public organisations, research groups, and individual researchers who want to improve the accessibility of their data for research and reproducibility by others. The goal of `metasyn` is to make it easy for data owners to share the structure and an approximation of the content of their data with others while keeping privacy concerns to a minimum.

With this goal in mind, `metasyn` distinguishes itself from existing software for generating synthetic data [e.g., @nowok2016synthpop; @templ2017simulation; @ping2017datasynthesizer] by strictly limiting the statistical information from the real data in the synthetic data. `Metasyn` explicitly avoids generating synthetic data with high analytical validity; instead, the synthetic data has realistic structure and plausible values, but multivariate relations are omitted ("augmented plausible synthetic data"; [@bates2019ons]). Moreover, our system provides an __auditable and editable intermediate representation__ in the form of a `.json` metadata file from which new data can be synthesized. 

These choices enable the software to generate synthetic data with __privacy and disclosure guarantees__ through a plug-in system, recognizing that different data owners have different needs and definitions of privacy. A data owner can define under which conditions they would accept open distribution of their synthetic data --- be it based on differential privacy [@dwork2006differential], statistical disclosure control [@hundepool2012statistical], k-anonymity [@sweeney2002k], or another specific definition of privacy. As part of the initial release of `metasyn`, we publish a [plug-in](https://github.com/sodascience/metasyn-disclosure-control) following the disclosure control guidelines from Eurostat [@bond2015guidelines].

# Software features

At its core, `metasyn` has three main functions:

1. __Estimation__: Automatically select distributions and fit them to a properly formatted tabular dataset, optionally with additional privacy guarantees.
2. __(De)serialization__: Create an intermediate representation of the fitted model for auditing, editing, and exporting.
3. __Generation__: Generate new synthetic datasets based on a fitted model.

## Estimation
The generative model for multivariate datasets in `metasyn` makes the assumption of marginal independence: each column is considered separately, just as is done in e.g., naïve Bayes classifiers [@hastie2009elements]. Formally, this leads to the following generative model for the $K$-variate data $\mathbf{x}$:

\begin{equation} \label{eq:model}
p(\mathbf{x}) = \prod_{k = 1}^K p(x_k)
\end{equation}

There are many advantages to this naïve approach when compared to more advanced generative models: it is transparent and explainable, it is able to flexibly handle data of mixed types, and it is computationally scalable to high-dimensional datasets. 

Model estimation starts with an appropriately pre-processed data frame, meaning it is tidy [@wickham2014tidy], each column has the correct data type, and missing data are represented by a missing value. Internally, our software uses the `polars` data frame library [@vink2024polars], as it is performant, has consistent data types, and natively supports missing data (i.e., `null` values). A simple example source table could look like this (note that categorical data has the appropriate `cat` data type, not `str`):

```
┌─────┬────────┬─────┬────────┬──────────┐
│ ID  ┆ fruits ┆ B   ┆ cars   ┆ optional │
│ --- ┆ ---    ┆ --- ┆ ---    ┆ ---      │
│ i64 ┆ cat    ┆ i64 ┆ cat    ┆ i64      │
╞═════╪════════╪═════╪════════╪══════════╡
│ 1   ┆ banana ┆ 5   ┆ beetle ┆ 28       │
│ 2   ┆ banana ┆ 4   ┆ audi   ┆ 300      │
│ 3   ┆ apple  ┆ 3   ┆ beetle ┆ null     │
│ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2        │
│ 5   ┆ banana ┆ 1   ┆ beetle ┆ -30      │
└─────┴────────┴─────┴────────┴──────────┘
```

For each data type, a set of candidate distributions is fitted (see Table \autoref{tbl:dist}), and then `metasyn` selects the one with the lowest BIC [@neath2012bayesian]. For distributions where BIC computation is impossible (e.g., for the string data type) a pseudo-BIC is created that trades off fit and complexity of the underlying models.

Table: \label{tbl:dist} Candidate distributions associated with data types in the core `metasyn` package.

| Variable type | Example                | Candidate distributions                                            |
| :------------ | :--------------------- | :----------------------------------------------------------------- |
| categorical   | yes/no, country        | Categorical (Multinoulli), Constant                                |
| continuous    | 1.0, 2.1, ...          | Uniform, Normal, LogNormal, TruncatedNormal, Exponential, Constant |
| discrete      | 1, 2, ...              | Poisson, Uniform, Normal, TruncatedNormal, Categorical, Constant   |
| string        | A108, C122, some words | Regex, Categorical, Faker, FreeText, Constant                      |
| date/time     | 2021-01-13, 01:40:12   | Uniform, Constant                                                  |

From this table, the string distributions deserve special attention as they are not commonly encountered as probability distributions. Regex (regular expression) inference is performed on structured strings using the companion package [RegexModel](https://pypi.org/project/regexmodel/). It is able to automatically detect structure such as room numbers (A108, C122, B109), e-mail addresses, websites, and more, which it summarizes using a probabilistic variant of regular expressions. The FreeText distribution detects the language (using [lingua](https://pypi.org/project/lingua-language-detector/)) and randomly picks words from that language. The [Faker](https://pypi.org/project/Faker/) distribution can generate specific data types such as localized addresses, when pre-specified by the user. 

Generative model estimation with `metasyn` can be performed as follows:

```python
from metasyn import MetaFrame

mf = MetaFrame.fit_dataframe(df)
```

## Serialization and deserialization
After a fitted model object is created, `metasyn` allows it to be transparently stored in a human- and machine-readable `.json` file. This file can be considered as metadata: it contains dataset-level descriptive information as well as the following variable-level information:

```json
{
  "name": "fruits",
  "type": "categorical",
  "dtype": "Categorical(ordering='physical')",
  "prop_missing": 0.0,
  "distribution": {
    "implements": "core.multinoulli",
    "version": "1.0",
    "provenance": "builtin",
    "class_name": "MultinoulliDistribution",
    "unique": false,
    "parameters": {
      "labels": ["apple", "banana"],
      "probs": [0.4, 0.6]
    }
  },
  "creation_method": { "created_by": "metasyn" }
}
```

This `.json` can be manually audited, edited, and after exporting this file, an unlimited number of synthetic records can be created without incurring additional privacy risks. Serialization and deserialization with `metasyn` can be performed as follows:

```python
mf.export("fruits.json")

# then, audit and transfer json

mf_out = MetaFrame.from_json("fruits.json")
```

## Data generation

For each variable in a fitted or deserialized model object, `metasyn` can randomly sample synthetic datapoints. Data generation (or synthetization) in `metasyn` can be performed as follows:

```python
from metasyn import MetaFrame

df_syn = mf.synthesize(10)
```

This may result in the following `polars` data frame[^1]. Note that missing values in the `optional` column are appropriately reproduced as well.

```
shape: (10, 5)
┌─────┬────────┬─────┬────────┬──────────┐
│ ID  ┆ fruits ┆ B   ┆ cars   ┆ optional │
│ --- ┆ ---    ┆ --- ┆ ---    ┆ ---      │
│ i64 ┆ cat    ┆ i64 ┆ cat    ┆ i64      │
╞═════╪════════╪═════╪════════╪══════════╡
│ 1   ┆ banana ┆ 4   ┆ beetle ┆ null     │
│ 2   ┆ banana ┆ 3   ┆ audi   ┆ null     │
│ …   ┆ …      ┆ …   ┆ …      ┆ …        │
│ 9   ┆ banana ┆ 4   ┆ beetle ┆ -30      │
│ 10  ┆ banana ┆ 2   ┆ beetle ┆ 172      │
└─────┴────────┴─────┴────────┴──────────┘
```

[^1]: This `polars` dataframe can be easily converted to a `pandas` dataframe using `df_syn.to_pandas()`

# Plug-ins and automatic privacy
In addition to its core features, the `metasyn` package allows for plug-ins: packages that alter the distribution fitting behaviour. Through this system, privacy guarantees can be built into `metasyn` ([privacy plug-in template](https://github.com/sodascience/metasyn-privacy-template)) and additional distributions can be supported ([distribution plug-in template](https://github.com/sodascience/metasyn-distribution-template)). The [`metasyn-disclosure-control`](https://github.com/sodascience/metasyn-disclosure-control) plug-in implements output guidelines from Eurostat [@bond2015guidelines] by including micro-aggregation. In this way, information transfer from the sensitive real data to the synthetic public data can be further limited. Disclosure control is done as follows:

```python
from metasyn import MetaFrame
from metasyncontrib.disclosure import DisclosurePrivacy

mf = MetaFrame.fit_dataframe(df, privacy=DisclosurePrivacy())
```

# Acknowledgements

This research was conducted in whole or in part using ODISSEI, the Open Data Infrastructure for Social Science and Economic Innovations (https://ror.org/03m8v6t10)

The `metasyn` project is supported by the FAIR Research IT Innovation Fund of Utrecht University (March 2023) 

# References