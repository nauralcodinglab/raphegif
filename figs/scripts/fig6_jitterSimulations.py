#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./src/')
sys.path.append('./figs/scripts/')

from jitterGIF import jitterGIF
import pltools


#%% INITIALIZE JITTERGIF

masterJGIF = jitterGIF(0.1)

masterJGIF.El = -60
masterJGIF.gl = 0.001
masterJGIF.C = 0.070

masterJGIF.gbar_K1 = 0
masterJGIF.gbar_K2 = 0

masterJGIF.Vthresh = -45
masterJGIF.Vreset = -75

masterJGIF.m_Vhalf = -23.7
masterJGIF.m_k = 0.10
masterJGIF.m_tau = 1.

masterJGIF.h_Vhalf = -76.0
masterJGIF.h_k = -0.11
masterJGIF.h_tau = 50.

masterJGIF.n_Vhalf = -24.2
masterJGIF.n_k = 0.20
masterJGIF.n_tau = 100.

masterJGIF.E_K = -101.


#%% PERFORM SIMULATIONS

# Simulation parameters
duration = 800
arrival_time = 150
dt = 0.1
no_reps = 100

# Synaptic parameters
ampli = 0.010 #nA
no_syn = 10
tau_rise = 1
tau_decay = 15

# Vars to perturb
jitter_var = [10, 15]
gk2_var = [0.001, 0.015]


# Create dict to hold simulation output
sim_output = masterJGIF.multiSim(jitter_var, gk2_var, no_reps, duration,
                                 arrival_time, tau_rise, tau_decay, ampli,
                                 no_syn, verbose = True)


#%% MAKE MAIN FIGURE


xlim = (0, 400)

plt.figure(figsize = (15, 8))

plt.subplot2grid((3, 2), (0, 0))
plt.title('{}ms jitter'.format(jitter_var[0]))
x       = np.arange(0, duration, dt)
plt.plot(x, 1e3 * sim_output['sample_syn'][:, :, 0].sum(axis = 1), 'k-')
plt.xlim(xlim)
plt.ylim(-5, 80)
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', omit_x = True, anchor = (0.9, 0.3), remove_frame = False)
plt.yticks([])
plt.xticks([])
pltools.hide_border()
plt.ylabel('$\sum I_{{syn, i}}$', rotation = 'horizontal')


plt.subplot2grid((3, 2), (0, 1))
plt.title('{}ms jitter'.format(jitter_var[1]))
x       = np.arange(0, duration, dt)
plt.plot(x, 1e3 * sim_output['sample_syn'][:, :, 1].sum(axis = 1), 'k-')
plt.ylim(-5, 80)
plt.xlim(xlim)
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', omit_x = True, anchor = (0.9, 0.3), remove_frame = False)
plt.yticks([])
plt.xticks([])
pltools.hide_border()
plt.ylabel('$\sum I_{{syn, i}}$', rotation = 'horizontal')


plt.subplot2grid((3, 2), (1, 0))

x       = np.arange(0, duration, dt)
y0      = sim_output['Vsub'][:, :, 0, 0].mean(axis = 1)
y0_sd   = sim_output['Vsub'][:, :, 0, 0].std(axis = 1)
y1      = sim_output['Vsub'][:, :, 0, 1].mean(axis = 1)
y1_sd   = sim_output['Vsub'][:, :, 0, 1].std(axis = 1)

x_pos = 80
plt.text(x_pos, -44, 'Spike threshold $= -45$mV', horizontalalignment = 'center')
plt.axhline(-45, linestyle = '--', dashes = (10, 5), color = 'k', zorder = 1)
plt.text(x_pos, -59, '$V_m = -60$mV', horizontalalignment = 'center')
plt.axhline(-60, linestyle = '--', dashes = (10, 5), color = 'k', zorder = 2)

plt.fill_between(x, y0 - y0_sd, y0 + y0_sd, facecolor = 'gray', edgecolor = 'gray', alpha = 0.3, zorder = 3)
plt.plot(x, y0, label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[0]), color = 'k', zorder = 5)
plt.fill_between(x, y1 - y1_sd, y1 + y1_sd, facecolor = 'r', edgecolor = 'r', alpha = 0.3, zorder = 4)
plt.plot(x, y1, label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[1]), color = 'r', zorder = 6)

plt.xlim(xlim)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (0.9, 0.25), remove_frame = False)
plt.yticks([])
plt.xticks([])
pltools.hide_border()
plt.ylabel('$V$', rotation = 'horizontal')


plt.subplot2grid((3, 2), (1, 1))

x       = np.arange(0, duration, dt)
y0      = sim_output['Vsub'][:, :, 1, 0].mean(axis = 1)
y0_sd   = sim_output['Vsub'][:, :, 1, 0].std(axis = 1)
y1      = sim_output['Vsub'][:, :, 1, 1].mean(axis = 1)
y1_sd   = sim_output['Vsub'][:, :, 1, 1].std(axis = 1)

x_pos = 80
plt.text(x_pos, -44, 'Spike threshold $= -45$mV', horizontalalignment = 'center')
plt.axhline(-45, linestyle = '--', dashes = (10, 5), color = 'k', zorder = 1)
plt.text(x_pos, -59, '$V_m = -60$mV', horizontalalignment = 'center')
plt.axhline(-60, linestyle = '--', dashes = (10, 5), color = 'k', zorder = 2)

plt.fill_between(x, y0 - y0_sd, y0 + y0_sd, facecolor = 'gray', edgecolor = 'gray', alpha = 0.3, zorder = 3)
plt.plot(x, y0, label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[0]), color = 'k', zorder = 5)
plt.fill_between(x, y1 - y1_sd, y1 + y1_sd, facecolor = 'r', edgecolor = 'r', alpha = 0.3, zorder = 4)
plt.plot(x, y1, label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[1]), color = 'r', zorder = 6)

plt.xlim(xlim)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (0.9, 0.25), remove_frame = False)
plt.yticks([])
plt.xticks([])
pltools.hide_border()
plt.ylabel('$V$', rotation = 'horizontal')

plt.subplot2grid((3, 2), (2, 0))
x       = np.arange(0, duration, dt)
plt.plot(x, np.cumsum(sim_output['pspk'][:, 0, 0]), 'k-', label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[0]))
plt.plot(x, np.cumsum(sim_output['pspk'][:, 0, 1]), 'r-', label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[1]))

plt.ylim(-0.05, 1.05)
plt.xlim(xlim)
plt.ylabel('Cumulative spike probability')
pltools.hide_border('tr')
plt.legend()

plt.subplot2grid((3, 2), (2, 1))
x       = np.arange(0, duration, dt)
plt.plot(x, np.cumsum(sim_output['pspk'][:, 1, 0]), 'k-', label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[0]))
plt.plot(x, np.cumsum(sim_output['pspk'][:, 1, 1]), 'r-', label = '$\\bar{{g}}_{{k2}} = {}$'.format(gk2_var[1]))

plt.ylim(-0.05, 1.05)
plt.xlim(xlim)
plt.ylabel('Cumulative spike probability')
pltools.hide_border('tr')
plt.legend()

plt.show()
