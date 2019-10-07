#%% IMPORT MODULES

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

import sys
sys.path.append(os.path.join('analysis', 'gating'))
from gating_tools import *
from grr.cell_class import Cell, Recording


#%% LOAD DATA

# Load pre-fitted gating curves.
DATA_PATH = os.path.join('data', 'processed', 'gating')

pdata = {}
for fname in os.listdir(DATA_PATH):
    if 'pdata' in fname:
        with open(os.path.join(DATA_PATH, fname), 'rb') as f:
            pdata[fname.split('_')[0]] = pickle.load(f)
            f.close()
    else:
        continue

fittedpts = {}
for fname in os.listdir(DATA_PATH):
    if 'fittedpts' in fname:
        with open(os.path.join(DATA_PATH, fname), 'rb') as f:
            fittedpts[fname.split('_')[0]] = pickle.load(f)
            f.close()
    else:
        continue

with open(os.path.join(DATA_PATH, 'gating_params.dat'), 'rb') as f:
    params = pickle.load(f)
    f.close()

# Load a nice example trace.
GATING_PATH = './data/raw/5HT/gating/'
beautiful_gating = Cell().read_ABF([GATING_PATH + '18619018.abf',
                                      GATING_PATH + '18619019.abf',
                                      GATING_PATH + '18619020.abf'])
beautiful_gating = Recording(np.array(beautiful_gating).mean(axis = 0))
beautiful_gating.set_dt(0.1)


#%% MAKE FIGURE

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')
plt.style.use('./figs/scripts/bhrd/poster_mplrc.dms')

plt.figure(figsize = (16, 4.5))

spec = gs.GridSpec(1, 2)
spec_tr = gs.GridSpecFromSubplotSpec(2, 1, spec[:, 0], height_ratios = [1, 0.2], hspace = 0.05)

plt.subplot(spec_tr[0, :])
plt.title(r'\textbf{A} Sample trace', loc = 'left')
plt.plot(
    beautiful_gating.t_mat[0, :36000, ::3] * 1e-3,
    beautiful_gating[0, 24000:60000, ::3] * 1e-3,
    'k-', lw = 1,
)
plt.xticks([])
plt.ylim(plt.ylim()[0], 1.2)
plt.ylabel('$I$ (nA)')

plt.subplot(spec_tr[1, :])
plt.plot(
    beautiful_gating.t_mat[1, :36000, ::3] * 1e-3,
    beautiful_gating[1, 24000:60000, ::3],
    '-', color = 'gray', lw = 1,
)
plt.yticks([-25, -75], ['$-25$', '$-75$'])
plt.ylabel('$V$ (mV)')
plt.xlabel('Time (s)')

plt.subplot(spec[:, 1])
plt.title(r'\textbf{B} Gating curves', loc = 'left')
plt.plot(pdata['peakact'][1, :, 0], max_normalize(pdata['peakact'][0, :, 0]), 'b-', alpha = 0.6, label = 'Activation')
plt.plot(pdata['peakact'][1, :, 1:], max_normalize(pdata['peakact'][0, :, 1:]), 'b-', alpha = 0.6)
plt.plot(pdata['peakinact'][1, :, 0], max_normalize(pdata['peakinact'][0, :, 0]), 'g-', alpha = 0.6, label = 'Inactivation')
plt.plot(pdata['peakinact'][1, :, 1:], max_normalize(pdata['peakinact'][0, :, 1:]), 'g-', alpha = 0.6)
plt.plot(fittedpts['peakact'][1, :], fittedpts['peakact'][0, :], '--', lw = 4, dashes = (8, 5), color = 'gray')
plt.plot(fittedpts['peakinact'][1, :], fittedpts['peakinact'][0, :], '--', lw = 4, dashes = (8, 5), color = 'gray')
plt.legend()
plt.xlabel('$V$ (mV)')
plt.ylabel(r'$g/g_{\mathrm{max}}$')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig2_perithresh.png'))
