What is metasyn?
================

``Metasyn`` is a python package for generating synthetic data with a focus on maintaining privacy.
It is aimed at owners of sensitive datasets such as public organisations, research groups, and individual researchers who
want to improve the accessibility, reproducibility and reusability of their data. The goal of ``metasyn`` is to make it easy
for data owners to share the structure and approximation of the content of their data with others with fewer privacy concerns.

Use cases
---------

Below are a few possible use cases for metasyn:

- A researcher who cannot share their dataset because of privacy concerns, but shares a synthetic twin for reproducibility.
- A data provider wants to show a preview of the real data using synthetic twins of their datasets.
- Create synthetic data before you actually have real data so that the analysis scripts can be written and tested.

It is specifically **not** designed for creating highly accurate synthetic data, where all relationships between columns are reproduced.

Key features
------------
- **Easy**: Creating your first synthetic dataset shouldn't take more than 15 minutes with our :doc:`quickstart <quick_start>`.
- **Fast**: Creating synthetic data takes mere seconds for medium sized (~1000 rows) datasets.
- **Safe and Understandable Synthetic Data**: As little information as possible is retained from the original dataset, and you can inspect and understand exactly which information is used to create your synthetic data.
- **Flexible**: You can :doc:`adjust <improve_synth>` the synthetic data columns to your liking, create your own distributions, extension and more.
- **Data Type Support**: ``Metasyn`` supports generating synthetic data for a variety of common data types including ``categorical``, ``string``, ``integer``, ``float``, ``date``, ``time``, and ``datetime``.
- **Integration with Faker**: ``Metasyn`` integrates with the `faker <https://github.com/joke2k/faker>`__ package, a Python library for generating fake data such as names and emails. Allowing for more realistic synthetic data.
- **Structured String Detection**: ``Metasyn`` identifies structured strings within your dataset, which can include formatted text, codes, identifiers, or any string that follows a specific pattern.
- **Handling Unique Values**: ``Metasyn`` can identify and process variables with unique values or keys in the data, preserving their uniqueness in the synthetic dataset.


For more detail on how metasyn works, see our `paper <https://github.com/sodascience/metasyn/blob/main/docs/paper/paper.pdf>`_.
