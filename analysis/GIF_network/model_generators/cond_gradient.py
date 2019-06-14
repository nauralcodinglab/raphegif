"""Generate a suite of AugmentedGIF models with various ratios of gl/gA."""
#%% IMPORT MODULES

from __future__ import division

import os
import pickle
import warnings
import copy

import numpy as np

import src.GIF_network as gfn


#%% SET CONSTANTS

SER_MOD_PATH = os.path.join('data', 'models', '5HT')

# Constants for models.
DT = 0.1

NO_COND_VALS = 20
NO_REPS = 50


#%% LOAD GIF MODELS

with open(os.path.join(SER_MOD_PATH, 'serkgifs.lmod'), 'rb') as f:
    sergifs = pickle.load(f)
    f.close()


#%% CREATE TEMPLATE SERGIF

template_sergif = sergifs[0]  # Use as a basis for hyperparams (eg eta basis)

# Set coefficient values.
params = [
    'gl', 'El', 'C', 'gbar_K1', 'h_tau',
    'gbar_K2', 'Vr', 'Vt_star', 'DV'
]
for param in params:
    # Check that model actually has param.
    if getattr(sergifs[0], param, None) is None:
        warnings.warn(
            'Mod does not have attribute {}. Skipping.'.format(param),
            RuntimeWarning
        )
        continue

    tmp_param_ls = []
    for mod in sergifs:
        tmp_param_ls.append(getattr(mod, param))
    setattr(template_sergif, param, np.median(tmp_param_ls))
del param, params, tmp_param_ls

# Set kernels (AHP, spike-triggered thresh change)
for kernel in ['eta', 'gamma']:
    tmp_param_arr = []
    for mod in sergifs:
        tmp_param_arr.append(getattr(mod, kernel).getCoefficients())
    vars(template_sergif)[kernel].setFilter_Coefficients(
        np.median(tmp_param_arr, axis = 0)
    )
del kernel, tmp_param_arr

# Fitted sergif models not needed anymore.
del sergifs


#%% CREATE SERGIFS WITH GL/GA GRADIENT

ga_vals = (
    np.linspace(0.003, 0.001, NO_COND_VALS)
    / template_sergif.mInf(template_sergif.Vt_star)
)
gl_vals = np.linspace(0.001, 0.003, NO_COND_VALS)

grad_sergifs = []

for i in range(NO_COND_VALS):
    tmp_gif = copy.deepcopy(template_sergif)
    tmp_gif.gbar_K1 = ga_vals[i]
    tmp_gif.gl = gl_vals[i]
    for j in range(NO_REPS):
        grad_sergifs.append(tmp_gif)


#%% PLACE IN GIFNET MODEL AND SAVE

subsample_gifnet = gfn.GIFnet(
    name = 'Subsample GIFs',
    ser_mod = grad_sergifs,
    dt = DT
)

# Remove interpolated filters first to save on disk space.
subsample_gifnet.clear_interpolated_filters()

# Save to disk.
OUTPUT_PATH = os.path.join('data', 'models', 'GIF_network')
with open(os.path.join(OUTPUT_PATH, 'condgrad.mod'), 'wb') as f:
    pickle.dump(subsample_gifnet, f)
    f.close()
