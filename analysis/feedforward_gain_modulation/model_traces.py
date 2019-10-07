#%% IMPORT MODULES

from __future__ import division

import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import seaborn as sns

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
sys.path.append('./src')
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps

#%% LOAD GIFS

GIFs = {}

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'rb') as f:
    GIFs['ser'] = pickle.load(f)
    f.close()

with open('./figs/scripts/gaba_neurons/opt_gaba_GIFs.pyc', 'rb') as f:
    GIFs['gaba'] = pickle.load(f)
    f.close()


#%% GET MEDIAN GIF PARAMETERS

mGIFs = {
    'ser': AugmentedGIF(0.1),
    'gaba': GIF(0.1)
}

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

for celltype in mGIFs.keys():

    # Set coefficient values.
    for param in ['gl', 'El', 'C', 'gbar_K1', 'h_tau', 'gbar_K2', 'Vr', 'Vt_star', 'DV']:
        if getattr(GIFs[celltype][0], param, None) is None:
            print '{} mod does not have attribute {}. Skipping.'.format(celltype, param)
            continue

        tmp_param_ls = []
        for mod in GIFs[celltype]:
            tmp_param_ls.append(getattr(mod, param))
        setattr(mGIFs[celltype], param, np.median(tmp_param_ls))

    for kernel in ['eta', 'gamma']:
        tmp_param_arr = []
        for mod in GIFs[celltype]:
            tmp_param_arr.append(getattr(mod, kernel).getCoefficients())
        vars(mGIFs[celltype])[kernel].setFilter_Coefficients(np.median(tmp_param_arr, axis = 0))

    mGIFs[celltype].printParameters()
    mGIFs[celltype].plotParameters()

#%% SIMPLE CURRENT STEP SIMULATION

IMG_PATH = None#'./figs/ims/ff_drn/'
plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

mGIFs['ser'].gbar_K1 = 0.010
mGIFs['ser'].h_tau = 50.
mGIFs['ser'].gbar_K2 = 0.002
mGIFs['ser'].DV = 0.5

I = np.concatenate((np.zeros(1500), -0.04 * np.ones(3500), 0.02 * np.ones(30000)))

t, V, eta, v_T, spks = mGIFs['ser'].simulate(I, mGIFs['ser'].El)

spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.6])

plt.figure()

plt.subplot(spec[0, :])
plt.plot(t, I, '-', color = 'gray')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec[1, :])
plt.plot(t, V, 'k-')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$V$ (mV)')

plt.subplot(spec[2, :])
plt.plot(spks, np.zeros_like(spks), 'k|')
for i in range(20):
    t, _, _, _, spks = mGIFs['ser'].simulate(I, mGIFs['ser'].El)
    plt.plot(spks, (i + 1) * np.ones_like(spks), 'k|')
plt.xlim(t[0], t[-1])
plt.ylabel('Repeat no.')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'median_KGIF_both_gk_fixed.png')

plt.show()


#%% CURRENT STEP SIMULATION FOR GABA

t, V, eta, v_T, spks = mGIFs['gaba'].simulate(I, mGIFs['gaba'].El)

spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.6])

plt.figure()

plt.subplot(spec[0, :])
plt.plot(t, I, '-', color = 'gray')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec[1, :])
plt.plot(t, V, 'k-')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$V$ (mV)')

plt.subplot(spec[2, :])
plt.plot(spks, np.zeros_like(spks), 'k|')
for i in range(20):
    t, _, _, _, spks = mGIFs['gaba'].simulate(I, mGIFs['gaba'].El)
    plt.plot(spks, (i + 1) * np.ones_like(spks), 'k|')
plt.xlim(t[0], t[-1])
plt.ylabel('Repeat no.')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'median_gabaGIF.png')

plt.show()
