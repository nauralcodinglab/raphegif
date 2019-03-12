#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm

import sys
sys.path.append('./src/')


from jitterGIF import jitterGIF
import src.pltools as pltools


#%% INITIALIZE JITTERGIF

masterJGIF = jitterGIF(0.1)

masterJGIF.El = -70
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
arrival_time = 250
dt = 0.1
no_reps = 250

# Synaptic parameters
ampli = 0.010 #nA
no_syn = 7
tau_rise = 1
tau_decay = 30

# Vars to perturb
jitter_var = [80]
gk2_var = [0.001, 0.015]
El_var = [-70, -50]


# Create dict to hold simulation output
sim_output = masterJGIF.multiSim(jitter_var, gk2_var, El_var, no_reps, duration,
                                 arrival_time, tau_rise, tau_decay, ampli,
                                 no_syn, verbose = True)


#%% PERFORM SIMULATIONS FOR 3D PLOT

sim_for_3D = masterJGIF.multiSim([80], np.linspace(0.001, 0.020, 40), np.linspace(-75, -45, 40),
                                 10, duration, arrival_time, tau_rise, 10, ampli,
                                 no_syn, verbose = True)

# Subtract El
for e_ in range(len(sim_for_3D['Els'])):

    sim_for_3D['Vsub'][:, :, 0, :, e_] -= sim_for_3D['Els'][e_]


#%% MAKE MAIN FIGURE

IMG_PATH = './figs/ims/'

mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
mpl.rc('text', usetex = True)
mpl.rc('svg', fonttype = 'none')

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rc('axes', linewidth = 0.8)

plt.figure(figsize = (16, 16))

gk2_color = [(0.7, 0, 0), (0.9, 0.2, 0.2)]
spec = mpl.gridspec.GridSpec(5, 3, height_ratios = (4, 1, 4, 1, 8), width_ratios = (2, 2, 1),
left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0.6)

# Col 1: example traces
Vax = plt.subplot(spec[0, 0])
plt.title('\\textbf{{A1}} Simulated response to synaptic input', loc = 'left')
plt.axhline(El_var[0], color = 'k', linestyle = '--', dashes = (10, 10))
plt.text(320, El_var[0] + 0.2, '$V_m = {}$mV'.format(El_var[0]), ha = 'center')
t = np.arange(0, duration, dt)
plt.plot(t, sim_output['Vsub'][:, 0, 0, 0, 0], label = 'Lo $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[0]), color = gk2_color[0])
plt.plot(t, sim_output['Vsub'][:, 0, 0, 1, 0], label = 'Hi $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[1]), color = gk2_color[1])
plt.legend().set_zorder(1e4)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', anchor = (0.98, 0.25), bar_spacing = 0, text_spacing = (0.02, -0.02))

cmdax = plt.subplot(spec[1, 0])
plt.plot(t, 1e3 * sim_output['sample_syn'][:, :, 0, 0].sum(axis = 1), '-', color = 'gray', linewidth = 2)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (0.98, 0.1))

pltools.join_plots(Vax, cmdax)


Vax = plt.subplot(spec[2, 0])
plt.title('\\textbf{{B1}} Simulated response to synaptic input', loc = 'left')
plt.axhline(El_var[1], color = 'k', linestyle = '--', dashes = (10, 10))
plt.text(320, El_var[1] + 0.2, '$V_m = {}$mV'.format(El_var[1]), ha = 'center')
t = np.arange(0, duration, dt)
plt.plot(t, sim_output['Vsub'][:, 0, 0, 0, 1], label = 'Lo $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[0]), color = gk2_color[0])
plt.plot(t, sim_output['Vsub'][:, 0, 0, 1, 1], label = 'Hi $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[1]), color = gk2_color[1])
plt.legend().set_zorder(1e4)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', anchor = (0.98, 0.4), bar_spacing = 0, text_spacing = (0.02, -0.02))

cmdax = plt.subplot(spec[3, 0])
plt.plot(t, 1e3 * sim_output['sample_syn'][:, :, 0, 0].sum(axis = 1), '-', color = 'gray', linewidth = 2)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (0.98, 0.2))

pltools.join_plots(Vax, cmdax)


# Col 2: mean traces

def mean_pm_sd(x, y_arr, color, zorder, label = None):

    if label is None:
        label = ''


    y_mean = y_arr.mean(axis = 1)
    y_sd = y_arr.std(axis = 1)

    plt.fill_between(x, y_mean - y_sd, y_mean + y_sd, facecolor = color, edgecolor = color, alpha = 0.3, zorder = zorder)
    plt.plot(x, y_mean, color = color, zorder = 1000 + zorder, label = label)

Vax = plt.subplot(spec[0, 1])
plt.title('\\textbf{{A2}} Mean response to random inputs', loc = 'left')
plt.axhline(El_var[0], color = 'k', linestyle = '--', dashes = (10, 10))
plt.text(320, El_var[0] + 0.2, '$V_m = {}$mV'.format(El_var[0]), ha = 'center')
t = np.arange(0, duration, dt)
mean_pm_sd(
t, sim_output['Vsub'][:, :, 0, 0, 0],
color = gk2_color[0], zorder = 1,
label = 'Lo $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[0]))
mean_pm_sd(
t, sim_output['Vsub'][:, :, 0, 1, 0],
color = gk2_color[1], zorder = 2,
label = 'Hi $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[1]))
plt.legend().set_zorder(1e4)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', anchor = (0.98, 0.3), bar_spacing = 0, text_spacing = (0.02, -0.02))

cmdax = plt.subplot(spec[1, 1])
mean_pm_sd(
t, 1e3 * sim_output['sample_syn'][:, :, :, 0].sum(axis = 1),
color = 'gray', zorder = 1
)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (0.98, 0.1))

pltools.join_plots(Vax, cmdax)


Vax = plt.subplot(spec[2, 1])
plt.title('\\textbf{{B2}} Mean response to random inputs', loc = 'left')
plt.axhline(El_var[1], color = 'k', linestyle = '--', dashes = (10, 10))
plt.text(320, El_var[1] + 0.2, '$V_m = {}$mV'.format(El_var[1]), ha = 'center')
#plt.axhline(-45, color = 'k', linestyle = '--', dashes = (10, 10))
#plt.text(250, -44.8, 'Spike threshold $\\approx -45$mV', ha = 'center')
t = np.arange(0, duration, dt)
mean_pm_sd(
t, sim_output['Vsub'][:, :, 0, 0, 1],
color = gk2_color[0], zorder = 10,
label = 'Lo $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[0]))
mean_pm_sd(
t, sim_output['Vsub'][:, :, 0, 1, 1],
color = gk2_color[1], zorder = 11,
label = 'Hi $K_{{Slow}}$ ({}pS)'.format(1e3 * gk2_var[1]))
plt.legend().set_zorder(1e4)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', anchor = (0.98, 0.4), bar_spacing = 0, text_spacing = (0.02, -0.02))

cmdax = plt.subplot(spec[3, 1])
mean_pm_sd(
t, 1e3 * sim_output['sample_syn'][:, :, :, 0].sum(axis = 1),
color = 'gray', zorder = 1
)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (0.98, 0.1))

pltools.join_plots(Vax, cmdax)


# Bar charts of peak amplitude

def add_bar(x, y_arr, color):

    y_mean = y_arr.max(axis = 0).mean()
    y_sd = y_arr.max(axis = 0).std()

    plt.errorbar(x, y_mean, y_sd, ls = 'none', color = 'k', zorder = 0)
    plt.bar(x, y_mean, 0.5, color = color, zorder = 1)

    plt.xlim(-0.5, 1.5)

plt.subplot(spec[:2, 2])
plt.title('\\textbf{{A3}} Mean amplitude', loc = 'left')
add_bar(0, sim_output['Vsub'][:, :, 0, 0, 0] - El_var[0], gk2_color[0])
add_bar(1, sim_output['Vsub'][:, :, 0, 1, 0] - El_var[0], gk2_color[1])
plt.xticks([i for i in range(len(gk2_var))], ['{}pS'.format(int(1e3 * i)) for i in gk2_var])
plt.xlabel('$\\bar{{g}}_{{Kslow}}$')
plt.ylabel('Amplitude (mV)')
pltools.hide_border('tr')

plt.subplot(spec[2:4, 2])
plt.title('\\textbf{{B3}} Mean amplitude', loc = 'left')
add_bar(0, sim_output['Vsub'][:, :, 0, 0, 1] - El_var[1], gk2_color[0])
add_bar(1, sim_output['Vsub'][:, :, 0, 1, 1] - El_var[1], gk2_color[1])
plt.xticks([i for i in range(len(gk2_var))], ['{}pS'.format(int(1e3 * i)) for i in gk2_var])
plt.xlabel('$\\bar{{g}}_{{Kslow}}$')
plt.ylabel('Amplitude (mV)')
pltools.hide_border('tr')


# 3D plot
Z_amp = np.squeeze(sim_for_3D['Vsub'].max(axis = 0).mean(axis = 0)).T
Y_V = np.broadcast_to(sim_for_3D['Els'].reshape((-1, 1)), Z_amp.shape)
X_gk2s = np.broadcast_to(sim_for_3D['gk2s'], Z_amp.shape)

gs00 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, spec[4, :])

ax0 = plt.subplot(gs00[0, 0], projection = '3d')
plt.title('\\textbf{{C1}} Synaptic integration depends on $V_m$ and $\\bar{{g}}_{{Kslow}}$ \nthroughout the physiological range', loc = 'left')
ax0.plot_surface(1e3 * X_gk2s, Y_V, Z_amp, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
ax0.invert_yaxis()
ax0.set_xticks([0, 5, 10, 15, 20])
ax0.set_yticks([-50, -60, -70])
ax0.set_zticks([3, 3.5, 4, 4.5])
ax0.set_xlabel('$\\bar{{g}}_{{Kslow}}$ (pS)')
ax0.set_ylabel('$V_m$ (mV)')
ax0.set_zlabel('Mean amplitude (mV)')

ax0.xaxis.labelpad = 12
ax0.yaxis.labelpad = 12
ax0.zaxis.labelpad = 12


plt.subplot(gs00[0, 1])
plt.title('\\textbf{{C2}} $K_{{Slow}}$ attenuates synaptic inputs \nonly at depolarized potentials', loc = 'left')

plt.plot(
1e3 * sim_for_3D['gk2s'],
np.squeeze(sim_for_3D['Vsub'][:, :, 0, :, 6].max(axis = 0).mean(axis = 0)).T,
':', color = gk2_color[1], label = '$V_m = -70$mV'
)
plt.plot(
1e3 * sim_for_3D['gk2s'],
np.squeeze(sim_for_3D['Vsub'][:, :, 0, :, 19].max(axis = 0).mean(axis = 0)).T,
'--', color = gk2_color[1], label = '$V_m = -60$mV'
)
plt.plot(
1e3 * sim_for_3D['gk2s'],
np.squeeze(sim_for_3D['Vsub'][:, :, 0, :, -8].max(axis = 0).mean(axis = 0)).T,
'-', color = gk2_color[1], label = '$V_m = -50$mV'
)
plt.legend().set_zorder(1e4)
plt.xticks([0, 5, 10, 15, 20])
plt.xlabel('$\\bar{{g}}_{{Kslow}}$ (pS)')
plt.ylabel('Mean amplitude (mV)')
pltools.hide_border('tr')

plt.savefig(IMG_PATH + 'fig6_jitterSimulations.png', dpi = 300)
plt.show()
