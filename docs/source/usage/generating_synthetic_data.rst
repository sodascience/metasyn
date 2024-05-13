Generating Synthetic Data
=========================

Metasyn can **generate synthetic data** from any :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` object.

.. image:: /images/pipeline_generation_simple.png
   :alt: Synthetic Data Generation
   :align: center

The generated synthetic data, emulates the original data's format and plausibility at the individual record
level and attempts to reproduce marginal (univariate) distributions where possible.
Generated values are based on the observed distributions.
The frequency of missing values is also maintained in the synthetically-augmented dataset.

The generated data does **not** preserve any relationships between variables.

.. admonition:: Prerequisite

   Before synthetic data can be generated, a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` object must be :doc:`created </usage/generating_metaframes>` or :doc:`loaded </usage/exporting_metaframes>`.

To generate a synthetic dataset, simply call the :meth:`MetaFrame.synthesize(n) <metasyn.metaframe.MetaFrame.synthesize>` method on a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` object. This method takes a parameter `n` which represents the number of rows of data that should be generated. By default (when `n` is not provided), metasyn tries to generate as many rows as in the original dataset.

.. image:: /images/pipeline_generation_code.png
   :alt: Synthetic Data Generation With Code Snippet
   :align: center

The following code generates 5 rows of data based on a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` named ``mf``.

.. code-block:: python
   
   mf.synthesize(5)

Upon succesful execution of the :meth:`MetaFrame.synthesize(n)<metasyn.metaframe.MetaFrame.synthesize>` method, a `Polars DataFrame <https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/index.html>`_ will be returned.



.. admonition:: Command-Line Interface

   Though we generally recommend using the Python API for using metasyn, you can also generate synthetic data using the CLI. Instructions for doing so can be found in the :doc:`CLI documentation </usage/cli>`.