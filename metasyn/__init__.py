"""Metasyn: a package for creating synthetic datasets.

Metasyn has three main functionalities:

- Estimation: Metasyn can create a MetaFrame, from a dataset. A MetaFrame is essentially a fitted model that characterizes the structure of the original dataset without storing actual values. It captures individual distributions and features, enabling generation of synthetic data based on these MetaFrames and can be seen as (statistical) metadata.
- Serialization: Metasyn can export a MetaFrame into an easy to read GMF file, allowing users to audit, understand, and modify their data generation model.
- Generation: Metasyn can generate synthetic data based on a MetaFrame. The synthetic data produced solely depends on the MetaFrame, thereby maintaining a critical separation between the original sensitive data and the synthetic data generated.
"""

from importlib.metadata import version

from metasyn.demo.dataset import demo_file
from metasyn.distribution.base import metadist
from metasyn.metaframe import MetaFrame
from metasyn.var import MetaVar

__all__ = ["MetaVar", "MetaFrame", "demo_file", "metadist"]
__version__ = version("metasyn")
