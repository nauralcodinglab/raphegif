#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')

from grr.cell_class import Cell, Recording
from grr import pltools


#%% LOAD DATA

def detect_spks(vec, ref = 3, dt = 0.1, return_inds = False):
    """Detects spikes as positive-going zero-crossings.
    Removes duplicates detected during the refractory period.
    """

    ref_ind = int(ref / dt)

    above_thresh = vec > 0
    below_thresh = ~above_thresh

    rising_edges = above_thresh[1:] & below_thresh[:-1]

    spks = np.where(rising_edges)[0] + 1

    # Find duplicates
    if len(spks) >= 1:
        redundant_pts = np.where(np.diff(spks) <= ref_ind)[0] + 1
        spks = np.delete(spks, redundant_pts)

    if return_inds:
        return spks
    else:
        return spks * dt


DATA_PATH = './data/raw/figdata/'

recs = Cell().read_ABF([DATA_PATH + fname for fname in ['18n06008.abf', '18n02000.abf']])

ISI_ls = []

for rec in recs:

    ISI_ls_tmp = []

    for i in range(rec.shape[2]):

        spks = detect_spks(rec[0, :, i])

        ISI_ls_tmp.append(np.diff(spks))

    ISI_ls.append(ISI_ls_tmp)


#%% INSPEC_DATA

spec_outer = gs.GridSpec(2, 1, height_ratios = [2, 1])
spec_tr = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[0, :], height_ratios = [0.3, 1], hspace = 0)

for i in range(len(recs)):

    rec = recs[i]
    rec.set_dt(0.1)
    ISIs = ISI_ls[i]


    plt.figure()

    I_ax = plt.subplot(spec_tr[0, :])
    V_ax = plt.subplot(spec_tr[1, :])
    ISI_ax = plt.subplot(spec_outer[1, :])

    I_ax.plot(rec.t_mat[0, :, 0], rec[1, :, 0], '-', color = 'gray', lw = 0.7)

    for j in range(5):

        V_ax.plot(rec.t_mat[0, :, j], rec[0, :, j], 'k-', lw = 0.7)
        ISI_ax.plot(ISIs[j], 'ko-', markersize = 2)

plt.show()


#%% MAKE FIGURE

IMG_PATH = './figs/ims/defence/'

example_cell = 0
use_sweeps = 5

plt.style.use('./figs/scripts/defence/defence_mplrc.dms')

spec_outer = gs.GridSpec(2, 1, height_ratios = [2.5, 1], left = 0.1, bottom = 0.15, top = 0.95, right = 0.95, hspace = 0.4)
spec_tr = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[0, :], height_ratios = [0.2, 1], hspace = 0)

plt.figure(figsize = (5, 2.5))

plt.subplot(spec_tr[0, :])
plt.plot(
    recs[example_cell].t_mat[1, :, 0], recs[example_cell][1, :, 0],
    '-', color = 'gray', lw = 0.7
)
pltools.add_scalebar(
    y_units = 'pA', omit_x = True,
    anchor = (0.05, 0.3), y_label_space = -0.02
)

plt.subplot(spec_tr[1, :])
plt.plot(
    recs[example_cell].t_mat[0, :, 0] / 1e3, recs[example_cell][0, :, 0],
    'k-', lw = 0.7
)
pltools.add_scalebar(
    y_units = 'mV', x_units = 's',
    bar_space = 0, anchor = (0.05, 0.3), x_on_left = False, y_label_space = -0.02
)

plt.subplot(spec_outer[1, :])
plt.title('Spike frequency adaptation', loc = 'left')
for ISIs in ISI_ls:

    ISIs_arr = np.empty((30, 50))
    ISIs_arr[:, :] = np.nan
    for i, x in enumerate(ISIs):
        ISIs_arr[i, :len(x)] = x

    ISIs_arr_mean = np.nanmean(ISIs_arr, axis = 0)
    ISIs_arr_mean /= ISIs_arr_mean[0]

    plt.plot([i+1 for i in range(len(ISIs_arr_mean))], ISIs_arr_mean, 'ko-', color = 'gray', markeredgecolor = 'k', markerfacecolor = 'gray')

plt.axhline(1, color = 'k', ls = 'dashed', lw = 0.5, dashes = (10, 10))
plt.yticks([0, 1, 2], ['0', '1', '2'])
plt.ylim(0, 2.5)
plt.ylabel('Normalized ISI')
plt.xlabel('ISI number')
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'long_current_step.png')

plt.show()
