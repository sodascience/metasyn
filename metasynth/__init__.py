"""MetaSynth: a package for creating synthetic datasets.

One part concerns the creation of the statistical metadata from the
original data, while the other part creates a synthetic dataset from the
metadata.
"""

from importlib.metadata import version

from metasynth.dataset import MetaDataset
from metasynth.var import MetaVar
from metasynth.demo.dataset import demo_file

__all__ = ["MetaVar", "MetaDataset"]
__version__ = version("metasynth")
