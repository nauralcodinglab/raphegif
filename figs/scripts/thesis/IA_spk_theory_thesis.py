#%% IMPORT MODULES

from __future__ import division

import sys
sys.path.append('./analysis/spk_timing/IA_mod')


import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import IAmod
import src.pltools as pltools


#%% INITIALIZE MODELS & PERFORM SIMULATIONS

Vin = np.empty((3000, 600))
Vin[:1000, :] = 0
Vin[1000:, :] = 25

low_IA_mod = IAmod.IAmod(1, 1.5, 2)
hi_IA_mod = IAmod.IAmod(10, 1.5, 2)

low_IA_sim = IAmod.Simulation(low_IA_mod, -60, Vin)
hi_IA_sim = IAmod.Simulation(hi_IA_mod, -60, Vin)


gaprime_vals = np.linspace(0, 15, 600)
continuous_mean = np.empty(len(gaprime_vals))

for i, gaprime in enumerate(gaprime_vals):

    print '\rSimulating {:.1f}%'.format(100 * i / len(gaprime_vals)),

    tmp_mod = IAmod.IAmod(gaprime, 1.5, 0)
    tmp_sim = IAmod.Simulation(tmp_mod, -60, Vin[:, 0][:, np.newaxis])

    continuous_mean[i] = tmp_sim.get_spk_latencies()[0] - 1


#%% LOAD ENORMOUS JITTER SIMULATION

with open('./analysis/spk_timing/IA_mod/latency_data.pyc', 'rb') as f:
    d = pickle.load(f)

stds = d['stds']
tiled_ga_vec = d['tiled_ga_vec']
tiled_tau_h_vec = d['tiled_tau_h_vec']


#%% MAKE PLOT

neurons_to_show = 20
hi_IA_color = (0.2, 0.2, 0.9)
low_IA_color = (0.07, 0.07, 0.3)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/thesis/'

spec_outer = gs.GridSpec(2, 3, hspace = 0.5, height_ratios = [1, 0.6])
spec_low_traces = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[0, 0], height_ratios = [0.2, 1, 0.4])
spec_hi_traces = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[0, 1], height_ratios = [0.2, 1, 0.4])
spec_continuous = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[1, :], wspace = 0.4)

plt.figure(figsize = (6, 5))

low_ax = plt.subplot(spec_low_traces[0, :])
plt.title('\\textbf{{A1}} Low $I_A$', loc = 'left')
plt.plot(
    low_IA_sim.t_vec,
    Vin[:, 0],
    color = 'gray', lw = 0.5
)
plt.annotate('0mV', (0.5, 2), ha = 'center')
plt.annotate('+25mV', (3, 23), ha = 'right', va = 'top')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec_low_traces[1, :], sharex = low_ax)
plt.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
plt.axhline(-45, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
plt.plot(
    low_IA_sim.t_vec,
    low_IA_sim.V[:, :neurons_to_show],
    color = low_IA_color, lw = 0.5, alpha = 0.1
)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (0.2, 0.3))
plt.annotate('-60mV', (3, -59), ha = 'right')
plt.annotate('Spike threshold', (3, -44), ha = 'right')

plt.subplot(spec_low_traces[2, :], sharex = low_ax)
for i in range(neurons_to_show):
    first_spk = 1e-3 * np.min(np.where(low_IA_sim.spks[:, i])[0])
    plt.plot(
        first_spk,
        i,
        'k|', markersize = 2
    )
pltools.add_scalebar(
    x_units = '$\\tau_{{\mathrm{{mem}}}}$', omit_y = True, x_size = 1,
    anchor = (0.8, -0.05), x_label_space = -0.1
)


hi_ax = plt.subplot(spec_hi_traces[0, :])
plt.title('\\textbf{{A2}} High $I_A$', loc = 'left')
plt.plot(
    hi_IA_sim.t_vec,
    Vin[:, 0],
    color = 'gray', lw = 0.5
)
plt.annotate('0mV', (0.5, 2), ha = 'center')
plt.annotate('+25mV', (3, 23), ha = 'right', va = 'top')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec_hi_traces[1, :], sharex = hi_ax)
plt.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
plt.axhline(-45, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
plt.plot(
    hi_IA_sim.t_vec,
    hi_IA_sim.V[:, :neurons_to_show],
    color = hi_IA_color, lw = 0.5, alpha = 0.1
)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (0.2, 0.3))
plt.annotate('-60mV', (3, -59), ha = 'right')
plt.annotate('Spike threshold', (3, -44), ha = 'right')

plt.subplot(spec_hi_traces[2, :], sharex = hi_ax)
for i in range(neurons_to_show):
    spk_inds = np.where(hi_IA_sim.spks[:, i])[0]
    if len(spk_inds) < 1:
        continue
    first_spk = 1e-3 * np.min(spk_inds)
    plt.plot(
        first_spk,
        i,
        'k|', markersize = 2
    )
pltools.add_scalebar(
    x_units = '$\\tau_{{\mathrm{{mem}}}}$', omit_y = True,
    x_size = 1, anchor = (0.8, -0.05), x_label_space = -0.1
)

plt.subplot(spec_outer[0, 2])
plt.title('\\textbf{{A3}} Spike latency', loc = 'left')
plt.hist(
    low_IA_sim.get_spk_latencies() - 1, edgecolor = 'none', facecolor = low_IA_color,
    label = 'Low $I_A$'
)
plt.hist(
    hi_IA_sim.get_spk_latencies()[~np.isnan(hi_IA_sim.get_spk_latencies())] - 1,
    edgecolor = 'none', facecolor = hi_IA_color, alpha = 0.7, bins = 10,
    label = 'High $I_A$'
)
plt.xlim(0, plt.xlim()[1])
plt.ylim(0, plt.ylim()[1] * 1.3)
plt.xlabel('Time from stim onset ($\\tau_{{\mathrm{{mem}}}}$)')
pltools.hide_border('ltr')
plt.yticks([])
plt.legend()


plt.subplot(spec_continuous[0, 0])
plt.title('\\textbf{{B1}} Effect of $I_A$ on latency', loc = 'left')
plt.plot(
    gaprime_vals,
    continuous_mean,
    'k-'
)
ind = np.argmin(np.abs(gaprime_vals - 1))
plt.plot(
    gaprime_vals[ind],
    continuous_mean[ind],
    'o', color = low_IA_color, markeredgecolor = 'k', markersize = 8,
    label = 'Low $I_A$'
)
ind = np.argmin(np.abs(gaprime_vals - 10))
plt.plot(
    gaprime_vals[ind],
    continuous_mean[ind],
    'o', color = hi_IA_color, markeredgecolor = 'k', markersize = 8,
    label = 'High $I_A$'
)
plt.ylabel('Spike latency ($\\tau_{{\mathrm{{mem}}}}$)')
plt.xlabel('Relative $I_A$ ($g_A / g_l$)')
pltools.hide_border('tr')
plt.legend()

plt.subplot(spec_continuous[0, 1])
plt.title('\\textbf{{B2}} Effect of $I_A$ on jitter', loc = 'left')
plt.plot(
    d['tiled_ga_vec'][:, 16],
    d['stds'][:, 16],
    'k-'
)
ind = np.argmin(np.abs(d['tiled_ga_vec'][:, 16] - 1))
plt.plot(
    d['tiled_ga_vec'][ind, 16],
    d['stds'][ind, 16],
    'o', color = low_IA_color, markeredgecolor = 'k', markersize = 8,
    label = 'Low $I_A$'
)
ind = np.argmin(np.abs(d['tiled_ga_vec'][:, 16] - 10))
plt.plot(
    d['tiled_ga_vec'][ind, 16],
    d['stds'][ind, 16],
    'o', color = hi_IA_color, markeredgecolor = 'k', markersize = 8,
    label = 'High $I_A$'
)
plt.xlabel('Relative $I_A$ ($g_A / g_l$)')
plt.ylabel('Spike latency SD ($\\tau_{{\mathrm{{mem}}}}$)')
pltools.hide_border('tr')
plt.legend()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_jitter_theory.png', dpi = 300)

plt.show()
