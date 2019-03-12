# TODO: make cmd look nice for v-steps

#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./analysis/gating/')


from cell_class import Cell, Recording
import src.pltools as pltools


#%% LOAD DATA

# Load current steps file
curr_steps = Cell().read_ABF('./figs/figdata/17n23038.abf')[0]
v_steps = Cell().read_ABF('./figs/figdata/18125015.abf')[0]

# Import passive membrane parameter data.
params = pd.read_csv('data/DRN_membrane_parameters.csv')

params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)

#%%

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

IMG_PATH = './figs/ims/'

plt.figure(figsize = (16, 10))

spec = gridspec.GridSpec(4, 4, height_ratios = (1, 0.75, 0.25, 1), left = 0.05,
                         right = 0.95, top = 0.95, bottom = 0.1,
                         hspace = 0.4)

#grid_dims = (3, 4)
hist_color = 'gray'

plt.subplot(spec[0, :2])
plt.title('\\textbf{{A1}} Motivation', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[0, 2])
plt.title('\\textbf{{A2}} DRN inputs', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[0, 3])
plt.title('\\textbf{{A3}} DRN outputs', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[1:3, :2])
plt.title('\\textbf{{B}} Identification of DRN 5HT neurons', loc = 'left')
pltools.hide_ticks()

Iax = plt.subplot(spec[1, 2])
cmdax = plt.subplot(spec[2, 2])
pltools.join_plots(Iax, cmdax)
xlim = (2000, 4500)
Iax.set_title('\\textbf{{C1}} Spiking', loc = 'left')
sweeps_to_use = [0, 4, 9, 19]
Iax.set_xlim(xlim)
Iax.plot(
np.broadcast_to(np.arange(0, curr_steps.shape[1]/10, 0.1)[:, np.newaxis], (curr_steps.shape[1], len(sweeps_to_use))),
curr_steps[0, :, sweeps_to_use].T - curr_steps[0, 20000:20500, sweeps_to_use].mean(axis = 1),
'k-', linewidth = 2
)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', anchor = (0.95, 0.3), text_spacing = (0.02, -0.05), bar_spacing = 0, ax = Iax)

cmdax.set_xlim(xlim)
cmdax.plot(
np.broadcast_to(np.arange(0, curr_steps.shape[1]/10, 0.1)[:, np.newaxis], (curr_steps.shape[1], len(sweeps_to_use))),
curr_steps[1, :, sweeps_to_use].T - curr_steps[1, 20000:20500, sweeps_to_use].mean(axis = 1),
'k-', linewidth = 2
)
pltools.add_scalebar(y_units = 'pA', anchor = (0.95, 0.3), omit_x = True)

Vax = plt.subplot(spec[1, 3])
cmdax = plt.subplot(spec[2, 3])
pltools.join_plots(Vax, cmdax)
xlim = (1750, 2750)
Vax.set_title('\\textbf{{C2}} Voltage steps', loc = 'left')
Vax.set_xlim(xlim)
Vax.set_ylim(-100, 1200)
sweeps_to_use = [0, 3, 6, 9]
Vax.plot(
np.broadcast_to(np.arange(0, v_steps.shape[1]/10, 0.1)[:, np.newaxis], (v_steps.shape[1], len(sweeps_to_use))),
v_steps[0, :, sweeps_to_use].T,
'k-', linewidth = 2
)
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', anchor = (0.9, 0.5), text_spacing = (0.02, -0.05), bar_spacing = 0, ax = Vax)

cmdax.set_xlim(xlim)
cmdax.plot(
np.broadcast_to(np.arange(0, v_steps.shape[1]/10, 0.1)[:, np.newaxis], (v_steps.shape[1], len(sweeps_to_use))),
v_steps[1, :, sweeps_to_use].T,
'k-', linewidth = 2
)
pltools.hide_ticks(ax = cmdax)
pltools.hide_border(ax = cmdax)


### Passive membrane parameters subplots

# Leak conductance
ax = plt.subplot(spec[3, 0])
plt.title('\\textbf{{D1}} Leak conductance', loc = 'left')
plt.hist(1e3/params_5HT['R'], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$g_l$ (pS)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(1e3/params_5HT['R'])
plt.text(0.98, 0.98,
'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)
plt.text(0.5, 0.02,
'$N = {}$ cells'.format(len(1e3/params_5HT['R'])),
verticalalignment = 'bottom', horizontalalignment = 'center', transform = ax.transAxes)

# Capacitance
ax = plt.subplot(spec[3, 1])
plt.title('\\textbf{{D2}} Capacitance', loc = 'left')
plt.hist(params_5HT['C'], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$C_m$ (pF)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['C'])
plt.text(0.98, 0.98,
'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)
plt.text(0.5, 0.02,
'$N = {}$ cells'.format(len(params_5HT['C'])),
verticalalignment = 'bottom', horizontalalignment = 'center', transform = ax.transAxes)

# Membrane time constant
ax = plt.subplot(spec[3, 2])
plt.title('\\textbf{{D3}} Time constant', loc = 'left')
plt.hist(params_5HT['R'] * params_5HT['C'] * 1e-3, color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\\tau$ (ms)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['R'] * params_5HT['C'] * 1e-3)
plt.text(0.98, 0.98,
'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)
plt.text(0.5, 0.02,
'$N = {}$ cells'.format(len(params_5HT['R'] * params_5HT['C'] * 1e-3)),
verticalalignment = 'bottom', horizontalalignment = 'center', transform = ax.transAxes)

# Estimated resting membrane potential
ax = plt.subplot(spec[3, 3])
plt.title('\\textbf{{D4}} Equilibrium potential', loc = 'left')
plt.hist(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\hat{{E}}_l$ (mV)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])
plt.text(0.98, 0.98,
'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)
plt.text(0.5, 0.02,
'$N = {}$ cells'.format(len(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])),
verticalalignment = 'bottom', horizontalalignment = 'center', transform = ax.transAxes)

#plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)

plt.savefig(IMG_PATH + 'fig1_physiology.png', dpi = 300)
plt.show()
