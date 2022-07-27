.. MetaSynth documentation master file, created by
   sphinx-quickstart on Thu May 12 14:26:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MetaSynth's documentation!
=====================================

MetaSynth is a package for generating statistical metadata from Pandas dataframes and creating synthetic data from this metadata.

It can be used when private datasets cannot be shared directly. By creating a synthetic dataset,
the data is safeguarded against the risk of disclosure. On the scale of disclosure/utilitly as defined
by the `Office for National Statistics <https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsmethodologyworkingpaperseriesnumber16syntheticdatapilot>`_,
MetaSynth resides in the *Synthetic-augmented plausible* category. This means that individual columns
should bear statistical/univariate resemblence to the original dataset, but the relationships between
the columns in the dataset is lost.

To create a metadata file for your dataset, you need to create a Pandas dataframe. For a simple example
see the :doc:`quick_start`.


.. toctree::
   :maxdepth: 2
   :caption: MetaSynth:
   :hidden:

   quick_start
   developer
   api/metasynth


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
