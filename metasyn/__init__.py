"""Metasyn: a package for creating synthetic datasets.

One part concerns the creation of the statistical metadata from the
original data, while the other part creates a synthetic dataset from the
metadata.
"""

from importlib.metadata import version

from metasyn.demo.dataset import demo_file
from metasyn.distribution.base import metadist
from metasyn.metaframe import MetaFrame
from metasyn.var import MetaVar

__all__ = ["MetaVar", "MetaFrame", "demo_file", "metadist"]
__version__ = version("metasyn")
