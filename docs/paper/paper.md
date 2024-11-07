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
Synthetic data is a promising tool for improving the accessibility of datasets which are too sensitive to be shared publicly. To this end, we introduce `metasyn`, a Python package for generating synthetic data from tabular datasets. Unlike existing synthetic data generation software, `metasyn` is built on a simple generative model that omits multivariate information. This choice enables transparency and auditability, keeps information leakage to a minimum, and enables privacy guarantees through a plug-in system. While the analytical validity of the generated data is thus intentionally limited, its potential uses are broad, including exploratory analyses, code development and testing, and external communication and teaching [@vankesteren2024democratize].

![Logo of the `metasyn` project.](img/logo.svg)

# Statement of need

`Metasyn` is aimed at owners of sensitive datasets such as public organisations, research groups, and individual researchers who want to improve the accessibility of their data for research and reproducibility by others. The goal of `metasyn` is to make it easy for data owners to share the structure and an approximation of the content of their data with others while keeping privacy concerns to a minimum.

With this goal in mind, `metasyn` distinguishes itself from existing software for generating synthetic data [e.g., @nowok2016synthpop; @templ2017simulation; @ping2017datasynthesizer] by strictly limiting the statistical information from the real data in the synthetic data. `Metasyn` explicitly avoids generating synthetic data with high analytical validity; instead, the synthetic data has realistic structure and plausible values, but multivariate relations are omitted ("augmented plausible synthetic data"; [@bates2019ons]). Moreover, our system provides an __auditable and editable intermediate representation__ in the form of a `.json` metadata file from which new data can be synthesized. 

These choices enable the software to generate synthetic data with __privacy and disclosure guarantees__ through a plug-in system, recognizing that different data owners have different needs and definitions of privacy. A data owner can define under which conditions they would accept open distribution of their synthetic data --- be it based on differential privacy [@dwork2006differential], statistical disclosure control [@hundepool2012statistical], k-anonymity [@sweeney2002k], or another specific definition of privacy. As part of the initial release of `metasyn`, we publish a [plug-in](https://github.com/sodascience/metasyn-disclosure-control) following the disclosure control guidelines from Eurostat [@bond2015guidelines].

# Software features

At its core, `metasyn` has three main functions: __estimation__, to fit a model to a properly formatted tabular dataset; __generation__, to synthesize new datasets based on a fitted model; and __(de)serialization__, to create a file from the model for auditing, editing, and saving.

## Estimation
Model estimation starts with an appropriately pre-processed data frame, meaning it is tidy [@wickham2014tidy], each column has the correct data type, and missing data are represented by a missing value. Accordingly, `metasyn` is built on the `polars` data frame library [@vink2024polars]. As an example, the first records of the "hospital" data built into `metasyn` are printed below:

```
┌────────────┬───────────────┬───────────────┬──────┬──────┬───────────────┐
│ patient_id ┆ date_admitted ┆ time_admitted ┆ type ┆ age  ┆ hours_in_room │
│ ---        ┆ ---           ┆ ---           ┆ ---  ┆ ---  ┆ ---           │
│ str        ┆ date          ┆ time          ┆ cat  ┆ i64  ┆ f64           │
╞════════════╪═══════════════╪═══════════════╪══════╪══════╪═══════════════╡
│ A5909X0    ┆ 2024-01-01    ┆ 10:30:00      ┆ IVT  ┆ null ┆ 3.633531      │
│ B4025X2    ┆ 2024-01-01    ┆ 11:23:00      ┆ IVT  ┆ 59   ┆ 6.932891      │
│ B6999X2    ┆ 2024-01-01    ┆ 11:58:00      ┆ IVT  ┆ 77   ┆ 1.970654      │
│ B9525X2    ┆ 2024-01-01    ┆ 16:56:00      ┆ MYE  ┆ null ┆ 1.620047      │
│ …          ┆ …             ┆ …             ┆ …    ┆ …    ┆ …             │
└────────────┴───────────────┴───────────────┴──────┴──────┴───────────────┘
```

Note that categorical data are encoded as `cat` (not `str`) and missing data is represented by `null` values. Model estimation with `metasyn` is then performed as follows:

```python
from metasyn import MetaFrame
mf = MetaFrame.fit_dataframe(df_hospital)
```

The generative model in `metasyn` makes the simplifying assumption of _marginal independence_: each column is considered separately, similar to naïve Bayes classifiers [@hastie2009elements]. For each column, a set of candidate distributions is fitted (see \autoref{tbl:dist}), and then `metasyn` selects the one that fits best (usually having the lowest BIC [@neath2012bayesian]). Key advantages of this approach are transparency and explainability, flexibility in handling mixed data types, and computational scalability to high-dimensional datasets. 

Table: \label{tbl:dist} Candidate distributions associated with data types in the core `metasyn` package.

| Data type   | Candidate distributions                                            |
| :---------- | :----------------------------------------------------------------- |
| Categorical | Categorical, Constant                                              |
| Continuous  | Uniform, Normal, LogNormal, TruncatedNormal, Exponential, Constant |
| Discrete    | Poisson, Uniform, Normal, TruncatedNormal, Categorical, Constant   |
| String      | Regex, Categorical, Faker, FreeText, Constant                      |
| Date/time   | Uniform, Constant                                                  |

From this table, the string distributions deserve special attention as they are not common probability distributions. The regex (regular expression) distribution uses the package [`regexmodel`](https://pypi.org/project/regexmodel/) to automatically detect structure such as room numbers (A108, C122, B109), identifiers, e-mail addresses, or websites. The FreeText distribution detects the language (using [lingua](https://pypi.org/project/lingua-language-detector/)) and randomly picks words from that language. The [Faker](https://pypi.org/project/Faker/) distribution can generate specific data types such as localized names and addresses pre-specified by the user. 


## Data generation

After creating a `MetaFrame`, `metasyn` can randomly sample synthetic datapoints from it. This is done using the `synthesize()` method:

```python
df_syn = mf.synthesize(3)
```

This may result in the following data frame. Note that missing values in the `age` column are appropriately reproduced as well.

```
┌────────────┬───────────────┬───────────────┬──────┬──────┬───────────────┐
│ patient_id ┆ date_admitted ┆ time_admitted ┆ type ┆ age  ┆ hours_in_room │
│ ---        ┆ ---           ┆ ---           ┆ ---  ┆ ---  ┆ ---           │
│ str        ┆ date          ┆ time          ┆ cat  ┆ i64  ┆ f64           │
╞════════════╪═══════════════╪═══════════════╪══════╪══════╪═══════════════╡
│ B7906X1    ┆ 2024-01-04    ┆ 13:32:00      ┆ IVT  ┆ 37   ┆ 4.955418      │
│ B0553X2    ┆ 2024-01-02    ┆ 10:54:00      ┆ IVT  ┆ 39   ┆ 3.872872      │
│ A5397X7    ┆ 2024-01-03    ┆ 18:16:00      ┆ CAT  ┆ null ┆ 6.569082      │
└────────────┴───────────────┴───────────────┴──────┴──────┴───────────────┘
```


## Serialization and deserialization
`MetaFrame`s can also be transparently stored in a human- and machine-readable `.json` metadata file. This file contains dataset-level descriptive information as well as variable-level information. This `.json` can be manually audited, edited, and after saving this file, an unlimited number of synthetic records can be created without incurring additional privacy risks. Serialization and deserialization with `metasyn` is done using the `save()` and `load()` methods:

```python
mf.save("hospital_admissions.json")
mf_new = MetaFrame.load("hospital_admissions.json")
```

# Privacy
As a general principle, `metasyn` errs on the side of privacy by default, aiming to recreate the structure but not all content and relations in the source data. For example, take the following sensitive dataset where study participants state how they use drugs in daily life:

```
┌────────────────┬─────────────────────────────────┐
│ participant_id ┆ drug_use                        │
│ ---            ┆ ---                             │
│ str            ┆ str                             │
╞════════════════╪═════════════════════════════════╡
│ OOWJAHA4       ┆ I use marijuana in the evening… │
│ 8CA1RV4P       ┆ I occasionally take CBD to hel… │
│ FMSVAKPM       ┆ Prescription medication helps … │
│ …              ┆ …                               │
└────────────────┴─────────────────────────────────┘
```

When creating synthetic data for this example, the information in the open answers is removed, and using our standard `FreeText` distribution this information is replaced by words from the detected language (English):

```
┌────────────────┬─────────────────────────────────┐
│ participant_id ┆ drug_use                        │
│ ---            ┆ ---                             │
│ str            ┆ str                             │
╞════════════════╪═════════════════════════════════╡
│ ZQJZQAB7       ┆ Lawyer let sort her yet line e… │
│ 7KDLEL0S       ┆ Particularly third myself edge… │
│ QBZKGXC7       ┆ Put color against call researc… │
└────────────────┴─────────────────────────────────┘
```

Additionally, the `metasyn` package supports [plug-ins](https://github.com/sodascience/metasyn-privacy-template) which alter the estimation behaviour. Through this system, privacy guarantees can be built into `metasyn` and additional distributions can be supported. For example, [`metasyn-disclosure-control`](https://github.com/sodascience/metasyn-disclosure-control) implements output guidelines from Eurostat [@bond2015guidelines] through _micro-aggregation_.

# Acknowledgements

This research was conducted in whole or in part using ODISSEI, the Open Data Infrastructure for Social Science and Economic Innovations (https://ror.org/03m8v6t10)

`metasyn` was supported by the Utrecht University FAIR Research IT Innovation Fund (March 2023) 

# References