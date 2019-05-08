#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import sys
sys.path.append(os.path.join('analysis', 'gating'))
import src.pltools as pltools
from src.cell_class import Cell
from gating_tools import *


#%% LOAD DATA

DRN398_fnames = {
    'baseline': ['18n16013.abf'],
    'V-steps': ['18n16015.abf'],
}
DRN393_fnames = {
    'baseline': ['18n16000.abf'],
    'V-steps': ['18n16003.abf'],
}

DATA_PATH = os.path.join('data', 'GABA_cells')

DRN398_recs = {}

for key in DRN398_fnames.keys():
    DRN398_recs[key] = Cell().read_ABF([os.path.join(DATA_PATH, 'DRN398_firing_vsteps', fname) for fname in DRN398_fnames[key]])

DRN393_recs = {}

for key in DRN393_fnames.keys():
    DRN393_recs[key] = Cell().read_ABF([os.path.join(DATA_PATH, 'DRN393_firing_vsteps', fname) for fname in DRN393_fnames[key]])


#%% MAKE FIGURE

plt.style.use(os.path.join('figs', 'scripts', 'bhrd', 'poster_mplrc.dms'))

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')

tr_xrange = slice(15000, 30000)
vclamp_xrange = slice(25000, 60000)

bl_params_bursty = {
    'sweeps': [-2],
    'cols': ['k'],
    'alphas': [1]
}
bl_params_nonbursty = {
    'sweeps': [-1],
    'cols': ['k'],
    'alphas': [1]
}
vclamp_params = {
    'sweeps': [2, 6, 10],
    'cols': ['k', 'k', 'k'],
    'alphas': [1, 1, 1]
}

def plot_traces(rec, primary_ax, secondary_ax, param_dict, primary_channel = 0, secondary_channel = 1, dt = 0.1):

    sweeps  = param_dict['sweeps']
    cols    = param_dict['cols']
    alphas  = param_dict['alphas']

    if not all([len(sweeps) == len(x) for x in [cols, alphas]]):
        raise ValueError('sweeps, cols, and alphas not of identical lengths.')

    t_vec = np.arange(0, (rec.shape[1] - 0.5) * dt, dt)

    if primary_ax is not None:
        for sw, col, alph in zip(sweeps, cols, alphas):
            primary_ax.plot(
                t_vec, rec[primary_channel, :, sw],
                '-', color = col, lw = 1., alpha = alph
            )
    if secondary_ax is not None:
        for sw, col, alph in zip(sweeps, cols, alphas):
            secondary_ax.plot(
                t_vec, rec[secondary_channel, :, sw],
                '-', color = 'gray', lw = 1., alpha = alph
            )


spec_outer = gs.GridSpec(1, 2, wspace = 0.4, bottom = 0.2, left = 0.08, right = 0.95)
spec_burstcell = gs.GridSpecFromSubplotSpec(
    2, 2, spec_outer[:, 0],
    height_ratios = [1, 0.3], hspace = 0.05, wspace = 0.5
)
spec_delaycell = gs.GridSpecFromSubplotSpec(
    2, 2, spec_outer[:, 1],
    height_ratios = [1, 0.3], hspace = 0.05, wspace = 0.5
)

plt.figure(figsize = (16, 3.5))

### Plot of bursty cell.
bl_V_ax = plt.subplot(spec_burstcell[0, 0])
bl_V_ax.set_title('\\textbf{{A1}} Bursting', loc = 'left')
bl_V_ax.set_xticks([])
bl_V_ax.set_ylabel('$V$ (mV)')
bl_I_ax = plt.subplot(spec_burstcell[1, 0])
#bl_I_ax.set_yticks([])
bl_I_ax.set_ylabel('$I$ (pA)')
bl_I_ax.set_xlabel('Time (ms)')

plot_traces(DRN398_recs['baseline'][0][:, tr_xrange, :], bl_V_ax, bl_I_ax, bl_params_bursty)

vclamp_I_ax = plt.subplot(spec_burstcell[0, 1])
vclamp_I_ax.set_title('\\textbf{{A2}} Currents', loc = 'left')
vclamp_I_ax.set_xticks([])
vclamp_I_ax.set_ylabel('$I$ (pA)')
vclamp_V_ax = plt.subplot(spec_burstcell[1, 1])
vclamp_V_ax.set_xlabel('Time (ms)')
vclamp_V_ax.set_ylabel('$V$ (mV)')

plot_traces(DRN398_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_I_ax, vclamp_V_ax, vclamp_params)

vclamp_ins = inset_axes(vclamp_I_ax, '30%', '25%', loc = 'upper center')
plot_traces(DRN398_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_ins, None, vclamp_params)
vclamp_ins.set_xlim(90, 350)
vclamp_ins.set_ylim(-90, 200)
vclamp_ins.set_xticks([])
vclamp_ins.set_yticks([])
mark_inset(vclamp_I_ax, vclamp_ins, loc1 = 2, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

vclamp_I_ax.set_ylim(-100, vclamp_I_ax.get_ylim()[1])
vclamp_I_ax.set_yticks(vclamp_I_ax.get_yticks()[1:])

### Plot of non-bursty cell.
bl_V_ax = plt.subplot(spec_delaycell[0, 0])
bl_V_ax.set_title('\\textbf{{B1}} Delayed firing', loc = 'left')
bl_V_ax.set_xticks([])
bl_V_ax.set_ylabel('$V$ (mV)')
bl_I_ax = plt.subplot(spec_delaycell[1, 0])
#bl_I_ax.set_yticks([])
bl_I_ax.set_ylabel('$I$ (pA)')
bl_I_ax.set_xlabel('Time (ms)')

plot_traces(DRN393_recs['baseline'][0][:, tr_xrange, :], bl_V_ax, bl_I_ax, bl_params_nonbursty)

vclamp_I_ax = plt.subplot(spec_delaycell[0, 1])
vclamp_I_ax.set_title('\\textbf{{B2}} Currents', loc = 'left')
vclamp_I_ax.set_xticks([])
vclamp_I_ax.set_ylabel('$I$ (pA)')
vclamp_V_ax = plt.subplot(spec_delaycell[1, 1])
vclamp_V_ax.set_xlabel('Time (ms)')
vclamp_V_ax.set_ylabel('$V$ (mV)')

plot_traces(DRN393_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_I_ax, vclamp_V_ax, vclamp_params)

vclamp_ins = inset_axes(vclamp_I_ax, '30%', '25%', loc = 'upper center')
plot_traces(DRN393_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_ins, None, vclamp_params)
vclamp_ins.set_xlim(90, 350)
vclamp_ins.set_ylim(-30, 60)
vclamp_ins.set_yticks([])
vclamp_ins.set_xticks([])
mark_inset(vclamp_I_ax, vclamp_ins, loc1 = 2, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

vclamp_I_ax.set_ylim(-100, vclamp_I_ax.get_ylim()[1])
vclamp_I_ax.set_yticks(vclamp_I_ax.get_yticks()[1:])

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig4_somassociation.png'))
