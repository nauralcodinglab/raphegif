#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from grr.cell_class import Cell
import src.pltools as pltools


#%% LOAD DATA

DRN393_fnames = {
    'baseline': ['18n16000.abf'],
    'V-steps': ['18n16003.abf'],
}

DATA_PATH = './data/GABA_cells/DRN393_firing_vsteps/'

DRN393_recs = {}

for key in DRN393_fnames.keys():
    DRN393_recs[key] = Cell().read_ABF([DATA_PATH + fname for fname in DRN393_fnames[key]])


#%% INSPECT RECORDINGS

for key in DRN393_recs.keys():
    DRN393_recs[key][0].plot()


#%% MAKE FIGURE

IMG_PATH = './figs/ims/gaba_cells/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

tr_xrange = slice(15000, 30000)
vclamp_xrange = slice(25000, 60000)

bl_params = {
    'sweeps': [3, -2, -1],
    'cols': ['k', 'k', 'k'],
    'alphas': [1, 0.5, 0.3]
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
                '-', color = col, lw = 0.5, alpha = alph
            )
    if secondary_ax is not None:
        for sw, col, alph in zip(sweeps, cols, alphas):
            secondary_ax.plot(
                t_vec, rec[secondary_channel, :, sw],
                '-', color = 'gray', lw = 0.5, alpha = alph
            )


spec_outer = gs.GridSpec(2, 2, wspace = 0.4, hspace = 0.4, bottom = 0.07, right = 0.95)
spec_bl_tr = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[0, :], height_ratios = [1, 0.2], hspace = 0)
spec_vclamp = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[1, :], height_ratios = [1, 0.2], hspace = 0)

plt.figure()

plt.suptitle('\\textbf{{WC currents in a non-bursty SOM neuron (DRN393)}}')

bl_V_ax = plt.subplot(spec_bl_tr[0, :])
bl_V_ax.set_title('\\textbf{{A}} Current steps', loc = 'left')
bl_V_ax.set_xticks([])
bl_V_ax.set_ylabel('$V$ (mV)')
bl_I_ax = plt.subplot(spec_bl_tr[1, :])
bl_I_ax.set_yticks([])
bl_I_ax.set_xlabel('Time (ms)')

plot_traces(DRN393_recs['baseline'][0][:, tr_xrange, :], bl_V_ax, bl_I_ax, bl_params)


vclamp_I_ax = plt.subplot(spec_vclamp[0, :])
vclamp_I_ax.set_title('\\textbf{{B}} Voltage steps (TTX)', loc = 'left')
vclamp_I_ax.set_xticks([])
vclamp_I_ax.set_ylabel('$I$ (pA)')
vclamp_V_ax = plt.subplot(spec_vclamp[1, :])
vclamp_V_ax.set_xlabel('Time (ms)')
vclamp_V_ax.set_ylabel('$V$ (mV)')

plot_traces(DRN393_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_I_ax, vclamp_V_ax, vclamp_params)

vclamp_ins = inset_axes(vclamp_I_ax, '30%', '25%', loc = 'upper center')
plot_traces(DRN393_recs['V-steps'][0][:, vclamp_xrange, :], vclamp_ins, None, vclamp_params)
vclamp_ins.set_xlim(90, 350)
vclamp_ins.set_ylim(-30, 60)
vclamp_ins.set_yticks([])
mark_inset(vclamp_I_ax, vclamp_ins, loc1 = 2, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

vclamp_I_ax.set_ylim(-50, vclamp_I_ax.get_ylim()[1])
vclamp_I_ax.set_yticks(vclamp_I_ax.get_yticks()[1:])

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'non_bursty_gaba_currents.png')

plt.show()
