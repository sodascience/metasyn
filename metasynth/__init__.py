"""MetaSynth: a package for creating synthetic datasets.

One part concerns the creation of the statistical metadata from the
original data, while the other part creates a synthetic dataset from the
metadata.
"""

from metasynth.var import MetaVar
from metasynth.dataset import MetaDataset

from . import _version
__version__ = _version.get_versions()['version']
