"""Tools for constructing constrained models of neural circuits."""

# Model components and tools
from . import AEC  # AEC
from . import Filter, Filter_Rect, Filter_Exps  # Filters
from . import cell_class, Experiment, pltools, ReadIBW, Simulation, SpikeTrainComparator, Tools, Trace  # Tools

# Models
from . import AugmentedGIF, CalciumGIF, gGIF, GIF, iGIF, jitterGIF, resGIF, SplineGIF, SpikingModel, ThresholdModel  # Spiking models
from . import SubthreshGIF  # Subthreshold models
from . import GIF_network  # Network models


# Metadata
__version__ = '0.0.1'
__author__ = 'Emerson Harkin'
__email__ = 'emerson.f.harkin at gmail dot com'
