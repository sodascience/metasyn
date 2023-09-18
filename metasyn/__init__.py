"""Metasyn: a package for creating synthetic datasets.

One part concerns the creation of the statistical metadata from the
original data, while the other part creates a synthetic dataset from the
metadata.
"""

from importlib.metadata import version

from metasyn.metaframe import MetaFrame
from metasyn.var import MetaVar
from metasyn.demo.dataset import demo_file
from metasyn.distribution.base import metadist

__all__ = ["MetaVar", "MetaFrame", "demo_file", "metadist"]
__version__ = version("metasyn")
