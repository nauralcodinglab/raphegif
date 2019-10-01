#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
from FeedForwardDRN import SynapticKernel
import src.GIF_network as gfn


#%% SET CONSTANTS

SER_MOD_PATH = os.path.join('data', 'models', '5HT')
SOM_MOD_PATH = os.path.join('data', 'models', 'GABA')

# Constants for models.
DT = 0.1

PROPAGATION_DELAY = int(2 / DT)
GABA_KERNEL = SynapticKernel(
    'alpha', tau = 25, ampli = -0.005, kernel_len = 400, dt = DT
).centered_kernel

NO_SER_NEURONS = 600
NO_GABA_NEURONS = 400
CONNECTION_PROB = 10. / NO_GABA_NEURONS
np.random.seed(514)
CONNECTIVITY_MATRIX = (
    np.random.uniform(size = (NO_SER_NEURONS, NO_GABA_NEURONS)) < CONNECTION_PROB
).astype(np.int8)


#%% LOAD GIF MODELS

with open(os.path.join(SER_MOD_PATH, 'serkgifs.lmod'), 'rb') as f:
    sergifs = pickle.load(f)
    f.close()

with open(os.path.join(SOM_MOD_PATH, 'gaba_gifs.mod'), 'rb') as f:
    somgifs = pickle.load(f)
    f.close()


#%% SPECIFY GIFNET MODEL
"""Randomly subsample GIFs/KGIFs fitted to DRN neurons.
"""

np.random.seed(515)
subsample_gifnet = gfn.GIFnet(
    name = 'Subsample GIFs',
    ser_mod = np.random.choice(sergifs, NO_SER_NEURONS),
    gaba_mod = np.random.choice(somgifs, NO_GABA_NEURONS),
    propagation_delay = PROPAGATION_DELAY,
    gaba_kernel = GABA_KERNEL,
    connectivity_matrix = CONNECTIVITY_MATRIX,
    dt = DT
)

#%% SAVE GIFNET MODEL

# Remove interpolated filters first to save on disk space.
subsample_gifnet.clear_interpolated_filters()

# Save to disk.
OUTPUT_PATH = os.path.join('data', 'models', 'GIF_network')
with open(os.path.join(OUTPUT_PATH, 'subsample.mod'), 'wb') as f:
    pickle.dump(subsample_gifnet, f)
    f.close()
