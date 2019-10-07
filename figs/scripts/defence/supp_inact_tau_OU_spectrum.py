#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')
sys.path.append('./figs/scripts/defence')

from grr.cell_class import Cell, Recording
from grr import pltools
from stimgen import Stim


#%% LOAD DATA

FIGDATA_PATH = './figs/figdata/'

with open(FIGDATA_PATH + 'inactivation_fits.pyc', 'rb') as f:
    IA_inact = pickle.load(f)

IA_inact.keys()


#%% OU NOISE SIMULATIONS

fast_ou = Stim('fast ou')
fast_ou.generate_OU(5000, 0, 50, 30, 0.5, 5000)
fast_ou.plot()
filtered_ou_tr_f = fast_ou.simulate_RC(1000, 75, 0)
unfiltered_ou_tr_f = fast_ou.command.flatten()
t_ou_tr = fast_ou.time.flatten()

slow_ou = Stim('slow ou')
slow_ou.generate_OU(5000, 0, 200, 30, 0.5, 5000)
filtered_ou_tr_s = slow_ou.simulate_RC(1000, 75, 0)
unfiltered_ou_tr_s = slow_ou.command.flatten()

dsigma_per = 5000
T = 50 * dsigma_per
reps = 50

fft_f_cmd = np.empty((reps, int(T / 0.1)))
fft_f_filt = np.empty_like(fft_f_cmd)

fft_s_cmd = np.empty_like(fft_f_cmd)
fft_s_filt = np.empty_like(fft_f_cmd)

fft_f = np.fft.fftfreq(int(T/0.1), 0.0001)

long_fast_noise = Stim('long fast ou')
long_slow_noise = Stim('long slow ou')

for i in range(reps):

    print('rep {}'.format(i))

    long_fast_noise.generate_OU(T, 0, 50, 30, 0.5, dsigma_per)
    long_fast_cmd = long_fast_noise.command.flatten()
    filtered_lf = long_fast_noise.simulate_RC(1000, 75, 0, False, False)

    fft_f_cmd[i, :] = np.abs(np.fft.fft(long_fast_cmd))
    fft_f_filt[i, :] = np.abs(np.fft.fft(filtered_lf.flatten()))

    long_slow_noise.generate_OU(T, 0, 200, 30, 0.5, dsigma_per)
    long_slow_cmd = long_slow_noise.command.flatten()
    filtered_ls = long_slow_noise.simulate_RC(1000, 75, 0, False, False)

    fft_s_cmd[i, :] = np.abs(np.fft.fft(long_slow_cmd))
    fft_s_filt[i, :] = np.abs(np.fft.fft(filtered_ls.flatten()))



#%% MAKE FIGURE

IMG_PATH = './figs/ims/defence/'

example_cell = 9

plt.style.use('./figs/scripts/defence/defence_mplrc.dms')

tau_spec = gs.GridSpec(1, 2, width_ratios = [1, 0.2], wspace = 0.6)

plt.figure(figsize = (5, 3))

plt.subplot(tau_spec[:, 0])
plt.plot(
    IA_inact['traces'][example_cell][1],
    IA_inact['traces'][example_cell][0],
    'k-', lw = 0.8, label = '$I$'
)
plt.plot(
    IA_inact['fitted_curves'][example_cell][1],
    IA_inact['fitted_curves'][example_cell][0],
    'b--', lw = 2, dashes = (4, 2), label = 'Monoexponential fit'
)
plt.xlim(2600, 2800)
plt.ylim(-50, plt.ylim()[1])
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA',
    anchor = (0.3, 0.1), bar_space = 0, x_on_left = False
)
plt.legend()

plt.subplot(tau_spec[:, 1])
plt.ylabel('Inactivation $\\tau$ (ms)')
plt.ylim(0, 70)
sns.swarmplot(
    y = IA_inact['inactivation_taus'],
    color = (0.2, 0.2, 0.9), linewidth = 1, edgecolor = 'gray'
)
plt.xticks([])
pltools.hide_border('trb')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'supp_ia_tau.png')

plt.show()


#%%

ou_spec = plt.GridSpec(2, 2, top = 0.9, right = 0.95, left = 0.1, bottom = 0.15, hspace = 0, wspace = 0.2)

plt.figure(figsize = (5, 3))

plt.subplot(ou_spec[0, 0])
plt.title('Fast Ornstein-Uhlenbeck noise')
plt.plot(t_ou_tr / 1e3, unfiltered_ou_tr_f,
    '-', color = 'gray', lw = 0.7, label = 'Raw noise ($\\tau = 50$ms)'
)
plt.plot(t_ou_tr / 1e3, filtered_ou_tr_f,
    'r-', lw = 0.7, alpha = 0.9, label = 'Filtered ($\\tau = 75$ms)'
)
pltools.add_scalebar(x_units = 's', omit_y = True, anchor = (0.9, 0.1), x_label_space = 0.02)
plt.yticks([])
plt.xlabel('Time (s)')
plt.legend(loc = 'upper right')


plt.subplot(ou_spec[1, 0])
plt.axvline(1e3/IA_inact['inactivation_taus'].mean(), color = (0.2, 0.2, 0.9), ls = 'dashed', dashes = (4, 2), lw = 0.8)
plt.axvline(1e3/75, color = 'k', ls = 'dashed', dashes = (4, 2), lw = 0.8)
plt.loglog(
    fft_f[fft_f>0], fft_f_cmd.mean(axis = 0)[fft_f>0],
    '-', color = 'gray'
)
plt.loglog(
    fft_f[fft_f>0], fft_f_filt.mean(axis = 0)[fft_f>0],
    'r-', alpha = 0.9
)
plt.annotate(
    '$I_A$ inact.', (1e3/IA_inact['inactivation_taus'].mean(), 1e6),
    (0.95, 0.9), textcoords = 'axes fraction', ha = 'right', va = 'top',
    arrowprops = {'arrowstyle': '->'}
)
plt.annotate(
    'Membrane $\\tau$', (1e3/75, 1e2),
    (0.4, 0.3), textcoords = 'axes fraction', ha = 'right', va = 'center',
    arrowprops = {'arrowstyle': '->'}
)
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')


plt.subplot(ou_spec[0, 1])
plt.title('Slow Ornstein-Uhlenbeck noise')
plt.plot(t_ou_tr / 1e3, unfiltered_ou_tr_s,
    '-', color = 'gray', lw = 0.7, label = 'Raw noise ($\\tau = 200$ms)'
)
plt.plot(t_ou_tr / 1e3, filtered_ou_tr_s,
    'r-', lw = 0.7, alpha = 0.9, label = 'Filtered ($\\tau = 75$ms)'
)
pltools.add_scalebar(x_units = 's', omit_y = True, anchor = (0.9, 0.1), x_label_space = 0.02)
plt.yticks([])
plt.xlabel('Time (s)')
plt.legend(loc = 'upper right')


plt.subplot(ou_spec[1, 1])
plt.axvline(1e3/IA_inact['inactivation_taus'].mean(), color = (0.2, 0.2, 0.9), ls = 'dashed', dashes = (4, 2), lw = 0.8)
plt.axvline(1e3/75, color = 'k', ls = 'dashed', dashes = (4, 2), lw = 0.8)
plt.loglog(
    fft_f[fft_f>0], fft_s_cmd.mean(axis = 0)[fft_f>0],
    '-', color = 'gray'
)
plt.loglog(
    fft_f[fft_f>0], fft_s_filt.mean(axis = 0)[fft_f>0],
    'r-', alpha = 0.9
)
plt.gca().set_yticklabels([])
#plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'supp_ou_filtering.png')

plt.show()
