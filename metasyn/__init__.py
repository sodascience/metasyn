"""Metasyn: a package for creating synthetic datasets.

Metasyn has three main purposes:

1. Estimation: Metasyn can create a MetaFrame from a dataset.
A MetaFrame is metadata describing a table, augmented with statistical
information on the columns. It captures individual distributions and
features and enables generation of synthetic data based on it.

2. Serialization and deserialization: Metasyn can export a
MetaFrame into an easy to read GMF file. This allows users to audit,
understand, and modify their data generation model. These GMF files
can also be imported back into Metasyn to generate synthetic data.

3. Generation: Metasyn can generate synthetic data based on a MetaFrame.
The synthetic data produced solely depends on the MetaFrame, thereby
maintaining a critical separation between the original sensitive data and the
generated synthetic data.
"""

from importlib.metadata import version

from metasyn.demo.dataset import demo_dataframe, demo_file
from metasyn.distribution.base import metadist
from metasyn.metaframe import MetaFrame
from metasyn.var import MetaVar
from metasyn.varspec import VarSpec

__all__ = ["MetaVar", "MetaFrame", "demo_file", "demo_dataframe", "metadist", "VarSpec"]
__version__ = version("metasyn")
