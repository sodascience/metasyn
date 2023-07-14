What is MetaSynth?
=====================================

MetaSynth is a Python package designed to generate tabular synthetic data for rigorous code testing and reproducibility.

The package has two main functionalities. First, it allows for the creation of metadata from an input dataset. This metadata describes the overarching structure and traits of the dataset. Second, MetaSynth allows for generation of synthetic data that aligns with this metadata. Instead of relying on the original dataset, the synthetic data is produced using the metadata. This approach ensures that the synthetic dataset remains separate and independent from any sensitive source data. Researchers and data owners can leverage this capability to generate and share synthetic versions of their sensitive data, mitigating privacy concerns. Furthermore, this separation between metadata and original data promotes reproducibility, as the metadata file can be easily shared and used to generate consistent synthetic datasets.

Key Features
--------

-  **Metadata Generation**: MetaSynth allows the extraction of metadata
   from a dataset provided as a Polars or Pandas dataframe. Metadata
   includes key characteristics such as variable names, types, data
   types, the percentage of missing values, and distribution attributes.
-  **Synthetic Data Generation**: MetaSynth allows for the generation of
   a polars DataFrame with synthetic data that resembles the original
   data.
-  **GMF Standard**: MetaSynth exports and imports metadata according to the `Generative Metadata Format (GMF) <https://github.com/sodascience/generative_metadata_format>`__. This is a standard designed to be easy to read and understand. 
-  **Distribution Fitting**: MetaSynth allows for manual and automatic
   distribution fitting.
-  **Data Type Support**: MetaSynth supports generating synthetic data
   for a variety of common data types including ``categorical``,
   ``string``, ``integer``, ``float``, ``date``, ``time``, and
   ``datetime``.
-  **Integration with Faker**: MetaSynth integrates with the
   `faker <https://github.com/joke2k/faker>`__ package, a Python library
   for generating fake data such as names and emails. Allowing for more
   realistic synthetic data.
-  **Structured String Detection**: MetaSynth identifies structured
   strings within your dataset, which can include formatted text, codes,
   identifiers, or any string that follows a specific pattern.
-  **Handling Unique Values**: MetaSynth can identify and process
   variables with unique values or keys in the data, preserving their
   uniqueness in the synthetic dataset, which is crucial for generating
   synthetic data that maintains the characteristics of the original
   dataset.
