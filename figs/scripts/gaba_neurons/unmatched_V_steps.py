#%% IMPORT MODULES

from __future__ import division

import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.cell_class import Cell
import src.pltools as pltools


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

unmatched_cells = [IVCell('DRN406', V_fnames = ['18n22015.abf']),
                   IVCell('DRN395', V_fnames = ['18n16005.abf']),
                   IVCell('DRN394', V_fnames = ['18n16004.abf'])]

DATA_PATH = './data/GABA_cells/unmatched_V_steps/'

for ce in unmatched_cells:
    ce.data_path = DATA_PATH
    ce.load_files()


#%% INSPECT RECORDINGS

cell_no = 0

unmatched_cells[cell_no].V_recs[0].plot()


#%% MAKE FIGURE

IMG_PATH = './figs/ims/gaba_cells/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

vclamp_xrange = slice(25000, 45000)

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

spec_outer = gs.GridSpec(len(unmatched_cells), 1)

plt.figure(figsize = (6, 6))

Letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i, ce in enumerate(unmatched_cells):

    spec_V_tmp = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[i, :], height_ratios = [1, 0.2], hspace = 0)

    V_Iax_tmp = plt.subplot(spec_V_tmp[0, :])
    V_Iax_tmp.set_title(r'\textbf{{{}}} {} voltage steps'.format(Letters[i], ce.name), loc = 'left')
    V_Iax_tmp.set_xticks([])
    V_Iax_tmp.set_ylabel('$I$ (pA)')
    V_Iax_tmp.set_ylim(-90, 370)
    pltools.hide_border('trb', V_Iax_tmp)
    V_Vax_tmp = plt.subplot(spec_V_tmp[1, :])
    V_Vax_tmp.set_yticks([])
    pltools.hide_border('ltr', V_Vax_tmp)
    if i == (len(unmatched_cells) - 1):
        V_Vax_tmp.set_xlabel('Time (ms)')
    else:
        V_Vax_tmp.set_xticks([])
        pltools.hide_border('b', V_Vax_tmp)

    plot_traces(ce.V_recs[0][:, vclamp_xrange, :], V_Iax_tmp, V_Vax_tmp, vclamp_params)

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'unmatched_v_steps.png')

plt.show()
