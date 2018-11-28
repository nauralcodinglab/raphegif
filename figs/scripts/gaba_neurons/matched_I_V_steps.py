#%% IMPORT MODULES

from __future__ import division

import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import sys
sys.path.append('./analysis/gating')
sys.path.append('./figs/scripts')
from cell_class import Cell
import pltools


#%% LOAD DATA

class IVCell(object):

    def __init__(self, name, I_fnames = None, V_fnames = None, data_path = './'):
        self.name = name
        self.I_fnames = I_fnames
        self.V_fnames = V_fnames
        self.data_path = data_path

        self.I_recs = None
        self.V_recs = None

    def load_files(self):
        if self.I_fnames is not None:
            self.I_recs = Cell().read_ABF([self.data_path + fname for fname in self.I_fnames])
        if self.V_fnames is not None:
            self.V_recs = Cell().read_ABF([self.data_path + fname for fname in self.V_fnames])


matched_cells = [IVCell('DRN399', ['18n22002.abf'], ['18n22000.abf', '18n22001.abf']),
                 IVCell('DRN400', ['18n22004.abf'], ['18n22003.abf']),
                 IVCell('DRN401', ['18n22006.abf', '18n22007.abf'], ['18n22005.abf']),
                 IVCell('DRN403', ['18n22011.abf', '18n22012.abf'], ['18n22008.abf', '18n22009.abf']),
                 IVCell('DRN405', ['18n22014.abf'], ['18n22013.abf']),
                 IVCell('DRN408', ['18n22019.abf'], ['18n22017.abf'])]


#%%
DATA_PATH = 'data/GABA_cells/matched_I_V_steps/'

for ce in matched_cells:
    ce.data_path = DATA_PATH
    ce.load_files()


#%% INSPECT RECORDINGS

cell_no = 5

matched_cells[cell_no].I_recs[0].plot()
matched_cells[cell_no].V_recs[0].plot()


#%% MAKE FIGURE

IMG_PATH = './figs/ims/gaba_cells/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

tr_xrange = slice(15000, 30000)
vclamp_xrange = slice(25000, 45000)

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

spec_outer = gs.GridSpec(len(matched_cells), 2)

plt.figure(figsize = (6, 8))

Letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i, ce in enumerate(matched_cells):

    spec_I_tmp = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[i, 0], height_ratios = [1, 0.2], hspace = 0)
    spec_V_tmp = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[i, 1], height_ratios = [1, 0.2], hspace = 0)

    I_Vax_tmp = plt.subplot(spec_I_tmp[0, :])
    I_Vax_tmp.set_title(r'\textbf{{{}1}} {} current steps'.format(Letters[i], ce.name), loc = 'left')
    I_Vax_tmp.set_xticks([])
    I_Vax_tmp.set_ylabel('$V$ (mV)')
    I_Vax_tmp.set_ylim(-120, -20)
    pltools.hide_border('trb', I_Vax_tmp)
    I_Iax_tmp = plt.subplot(spec_I_tmp[1, :])
    I_Iax_tmp.set_yticks([])
    pltools.hide_border('ltr', I_Iax_tmp)
    if i == (len(matched_cells) - 1):
        I_Iax_tmp.set_xlabel('Time (ms)')
    else:
        I_Iax_tmp.set_xticks([])
        pltools.hide_border('b', I_Iax_tmp)

    plot_traces(ce.I_recs[0][:, tr_xrange, :], I_Vax_tmp, I_Iax_tmp, bl_params)


    V_Iax_tmp = plt.subplot(spec_V_tmp[0, :])
    V_Iax_tmp.set_title(r'\textbf{{{}2}} {} voltage steps'.format(Letters[i], ce.name), loc = 'left')
    V_Iax_tmp.set_xticks([])
    V_Iax_tmp.set_ylabel('$I$ (pA)')
    V_Iax_tmp.set_ylim(-120, 610)
    pltools.hide_border('trb', V_Iax_tmp)
    V_Vax_tmp = plt.subplot(spec_V_tmp[1, :])
    V_Vax_tmp.set_yticks([])
    pltools.hide_border('ltr', V_Vax_tmp)
    if i == (len(matched_cells) - 1):
        V_Vax_tmp.set_xlabel('Time (ms)')
    else:
        V_Vax_tmp.set_xticks([])
        pltools.hide_border('b', V_Vax_tmp)

    plot_traces(ce.V_recs[0][:, vclamp_xrange, :], V_Iax_tmp, V_Vax_tmp, vclamp_params)

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'I_V_various_cells.png')

plt.show()
