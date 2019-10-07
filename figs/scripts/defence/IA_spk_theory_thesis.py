#%% IMPORT MODULES

from __future__ import division

import sys
sys.path.append('./analysis/spk_timing/IA_mod')
sys.path.append('./figs/scripts')

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import IAmod
import pltools


#%% INITIALIZE MODELS & PERFORM SIMULATIONS

Vin = np.empty((3000, 600))
Vin[:1000, :] = 0
Vin[1000:, :] = 25

low_IA_mod = IAmod.IAmod(1, 1.5, 2)
hi_IA_mod = IAmod.IAmod(10, 1.5, 2)

low_IA_sim = IAmod.Simulation(low_IA_mod, -60, Vin)
hi_IA_sim = IAmod.Simulation(hi_IA_mod, -60, Vin)



#%% MAKE PLOT

neurons_to_show = 20
hi_IA_color = (0.2, 0.2, 0.9)
low_IA_color = (0.07, 0.07, 0.3)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/defence/'

spec_outer = gs.GridSpec(1, 3, hspace = 0.5, left = 0.05, right = 0.92)
spec_low_traces = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[0, 0], height_ratios = [0.2, 1, 0.4])
spec_hi_traces = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[0, 1], height_ratios = [0.2, 1, 0.4])

plt.figure(figsize = (6, 3))

low_ax = plt.subplot(spec_low_traces[0, :])
plt.title('\\textbf{{A}} Low $I_A$ (1$\\times g_l$)', loc = 'left')
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
plt.title('\\textbf{{B}} High $I_A$ (10$\\times g_l$)', loc = 'left')
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
plt.title('\\textbf{{C}} Spike latency', loc = 'left')
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


if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_jitter_theory.png', dpi = 300)

plt.show()
