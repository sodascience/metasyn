"""MetaSynth: a package for creating synthetic datasets.

One part concerns the creation of the statistical metadata from the
original data, while the other part creates a synthetic dataset from the
metadata.
"""

from importlib.metadata import version

from metasynth.dataset import MetaFrame
from metasynth.var import MetaVar
from metasynth.spec import MetaFrameSpec
from metasynth.demo.dataset import demo_file
from metasynth.distribution.base import metadist

__all__ = ["MetaVar", "MetaFrame", "demo_file", "metadist", "MetaFrameSpec"]
__version__ = version("metasynth")
