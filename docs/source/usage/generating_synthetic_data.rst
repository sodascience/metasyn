Generating Synthetic Data
=========================

MetaSynth can **generate synthetic data** from any :obj:`MetaDataset <metasynth.dataset.MetaDataset>`.

.. image:: /images/flow_synthetic_data_generation.png
   :alt: Synthetic_data_generation

The generated synthetic data, emulates the original data's format and
plausibility at the individual record level and attempts to reproduce
marginal (univariate) distributions where possible. Generated values are
based on the observed distributions while adding a degree of variance
and smoothing. The frequency of missing values is also maintained in the synthetically-augmented dataset.

The generated data does **not** aim to preserve the
relationships between variables.


Prerequisites
-------------
Before synthetic data can be generated, a :obj:`MetaDataset <metasynth.dataset.MetaDataset>` must be created or loaded (see :doc:`/usage/generating_metadata` for instructions). 

Generating a synthetic dataset
-------------------------
To generate a synthetic dataset, call the :meth:`MetaDataset.synthesize(n)<metasynth.dataset.MetaDataset.synthesize>` method on a :obj:`MetaDataset <metasynth.dataset.MetaDataset>` object.
This method takes an integer parameter `n` which represents the number of rows of data that should be generated. This parameter *must* be specified when calling the method.
Upon succesful execution of the :meth:`MetaDataset.synthesize(n)<metasynth.dataset.MetaDataset.synthesize>` method, a `Pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ will be returned.

**In-code example**:

.. code-block:: python

   metadataset.synthesize(5)
..



