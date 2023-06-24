Generating Synthetic Data
=========================

MetaSynth can **generate synthetic data** from any :obj:`MetaDataset <metasynth.dataset.MetaDataset>`.

.. image:: /images/flow_synthetic_data_generation.png
   :alt: Synthetic_data_generation

The generated synthetic data, emulates the original data's format and
plausibility at the individual record level and attempts to reproduce
marginal (univariate) distributions where possible. Generated values are
based on the observed distributions while adding a degree of variance
and smoothing. The generated data does **not** aim to preserve the
relationships between variables. The frequency of missing values and
their codes are maintained in the synthetically-augmented dataset.



Prerequisites
-------------
Before synthetic data can be generated, a :obj:`MetaDataset <metasynth.dataset.MetaDataset>` must be created or loaded (see :doc:`/usage/generating_metadata` for instructions). 


Example
-------
Here's an example of how to generate synthetic data from a metadataset:

.. code-block:: python

   import metasynth

    // example here
