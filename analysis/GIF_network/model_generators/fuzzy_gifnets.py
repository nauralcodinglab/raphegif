"""Create models for GIF-based DRN network with hand-massaged parameters.
Goal is to create network models ideal to implement gain modulation,
such that if even these networks can't do it, the DRN definitely can't.

Models are placed in MODEL_PATH
"""

#%% IMPORT MODULES

from __future__ import division

import pickle
from copy import deepcopy
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
import src.GIF_network as gfn
from FeedForwardDRN import SynapticKernel
from src.GIF import GIF
from src.AugmentedGIF import AugmentedGIF
from src.Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from src.Filter_Exps import Filter_Exps


#%% SET CONSTANTS

# Location to deposit generated models.
MODEL_PATH = os.path.join('data', 'models', 'GIF_network')

# Constants for models.
DT = 0.1

PROPAGATION_DELAY = int(2 / DT)
GABA_KERNEL = SynapticKernel(
    'alpha', tau = 25, ampli = -0.001, kernel_len = 400, dt = DT
).centered_kernel

NO_SER_NEURONS = 1200
NO_GABA_NEURONS = 800
CONNECTION_PROB = 25. / NO_GABA_NEURONS
CONNECTIVITY_MATRIX = (
    np.random.uniform(size = (NO_SER_NEURONS, NO_GABA_NEURONS)) < CONNECTION_PROB
).astype(np.int8)



#%% CREATE GIFNETs

#First, create models based on median GIF/AugmentedGIF parameters.
GIFs = {}

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'rb') as f:
    GIFs['ser'] = pickle.load(f)
    f.close()

with open('./figs/scripts/gaba_neurons/opt_gaba_GIFs.pyc', 'rb') as f:
    GIFs['gaba'] = pickle.load(f)
    f.close()


# Create dict to hold models constrained to median param estimates.
mGIFs = {
    'ser': AugmentedGIF(0.1),
    'gaba': GIF(0.1)
}

# Set hyperparameters.
mGIFs['ser'].Tref = 6.5
mGIFs['ser'].eta = Filter_Rect_LogSpaced()
mGIFs['ser'].eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
mGIFs['ser'].gamma = Filter_Exps()
mGIFs['ser'].gamma.setFilter_Timescales([30, 300, 3000])

mGIFs['gaba'].Tref = 4.0
mGIFs['gaba'].eta = Filter_Rect_LogSpaced()
mGIFs['gaba'].eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
mGIFs['gaba'].gamma = Filter_Exps()
mGIFs['gaba'].gamma.setFilter_Timescales([30, 300, 3000])

# Extract coefficients for both celltypes.
for celltype in mGIFs.keys():

    mGIFs[celltype].param_sds = {} # Add a dict to hold uncertainties for parameters.

    # Set coefficient values.
    for param in ['gl', 'El', 'C', 'gbar_K1', 'h_tau', 'gbar_K2', 'Vr', 'Vt_star', 'DV']:
        print 'Extracting {} for {} median model.'.format(param, celltype)
        if getattr(GIFs[celltype][0], param, None) is None:
            warnings.warn(
                '{} mod does not have attribute {}. Skipping.'.format(celltype, param),
                RuntimeWarning
            )
            continue

        tmp_param_ls = []
        for mod in GIFs[celltype]:
            tmp_param_ls.append(getattr(mod, param))
        setattr(mGIFs[celltype], param, np.mean(tmp_param_ls))
        mGIFs[celltype].param_sds[param] = np.std(tmp_param_ls)

        IQR_tmp = np.percentile(tmp_param_ls, 75) - np.percentile(tmp_param_ls, 25)
        if np.std(tmp_param_ls) > (5. * IQR_tmp):
            warnings.warn(
                '{} mod param {} IQR << std ({} vs {})'.format(
                    celltype, param, np.std(tmp_param_ls), IQR_tmp
                )
            )

    for kernel in ['eta', 'gamma']:
        tmp_param_arr = []
        for mod in GIFs[celltype]:
            tmp_param_arr.append(getattr(mod, kernel).getCoefficients())
        vars(mGIFs[celltype])[kernel].setFilter_Coefficients(np.mean(tmp_param_arr, axis = 0))

del GIFs

# Wrap everything in a GIFnet model.
mGIF_mod = gfn.GIFnet(
    name = 'Fuzzy mean GIFs',
    ser_mod = mGIFs['ser'],
    gaba_mod = mGIFs['gaba'],
    propagation_delay = PROPAGATION_DELAY,
    gaba_kernel = GABA_KERNEL,
    connectivity_matrix = CONNECTIVITY_MATRIX,
    dt = DT
)

print 'Saving fuzzy mean GIFnet...'
with open(os.path.join(MODEL_PATH, 'fuzzy_mean_gifnet.mod'), 'wb') as f:
    pickle.dump(mGIF_mod, f)
    f.close()
del mGIF_mod
print 'Done fuzzy mean GIFnet!'


# GIFnet using 5HT cells with manually tuned params.
sergif_manual = deepcopy(mGIFs['ser'])

sergif_manual.gbar_K1 = 0.010
sergif_manual.h_tau = 50.
sergif_manual.gbar_K2 = 0.002
sergif_manual.DV = 2.5

for param in ['h_tau', 'gbar_K2', 'DV']:
    sergif_manual.param_sds[param] = 0.

sergif_manual.param_sds['gbar_K1'] = 0.001
sergif_manual.param_sds['gl'] = min(
    sergif_manual.gl * 0.2, sergif_manual.param_sds['gl']
)

fuzzy_sergif_manual_mod = gfn.GIFnet(
    name = '5HT manually tuned',
    ser_mod = sergif_manual,
    gaba_mod = mGIFs['gaba'],
    propagation_delay = PROPAGATION_DELAY,
    gaba_kernel = GABA_KERNEL,
    connectivity_matrix = CONNECTIVITY_MATRIX,
    dt = DT
)

print 'Saving fuzzy GIFnet with manually tuned 5HT params.'
with open(os.path.join(MODEL_PATH, 'fuzzy_manual_gifnet.mod'), 'wb') as f:
    pickle.dump(fuzzy_sergif_manual_mod, f)
    f.close()
del fuzzy_sergif_manual_mod
print 'Done fuzzy GIFnet with manually tuned 5HT params!'

# GIFnet using manually tuned 5HT cells with IA knocked out.

sergif_man_noIA = deepcopy(sergif_manual)
sergif_man_noIA.gbar_K1 = 0.
sergif_man_noIA.param_sds['gbar_K1'] = 0.

fuzzy_sergif_noIA_mod = gfn.GIFnet(
    name = '5HT manually tuned no IA',
    ser_mod = sergif_man_noIA,
    gaba_mod = mGIFs['gaba'],
    propagation_delay = PROPAGATION_DELAY,
    gaba_kernel = GABA_KERNEL,
    connectivity_matrix = CONNECTIVITY_MATRIX,
    dt = DT
)

print 'Saving fuzzy GIFnet with manually tuned 5HT params and IA KO.'
with open(os.path.join(MODEL_PATH, 'fuzzy_manual_noIA_gifnet.mod'), 'wb') as f:
    pickle.dump(fuzzy_sergif_noIA_mod, f)
    f.close()
del fuzzy_sergif_noIA_mod
print 'Done fuzzy GIFnet with manually tuned 5HT params and IA KO!'

