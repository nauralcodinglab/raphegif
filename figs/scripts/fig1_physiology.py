#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')

from cell_class import Cell, Recording
import pltools


#%% LOAD DATA

# Import passive membrane parameter data.
params = pd.read_csv('data/DRN_membrane_parameters.csv')

params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)

#%%

plt.figure(figsize = (14.67, 10))

grid_dims = (3, 4)
hist_color = 'gray'

plt.subplot2grid(grid_dims, (0, 0), colspan = 2)
plt.title('A1 Motivation', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid(grid_dims, (0, 2))
plt.title('A2 DRN inputs', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid(grid_dims, (0, 3))
plt.title('A3 DRN outputs', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid(grid_dims, (1, 0), colspan = 2)
plt.title('B Identification of DRN 5HT neurons', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid(grid_dims, (1, 2))
plt.title('C1 Spiking', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid(grid_dims, (1, 3))
plt.title('C2 Voltage steps', loc = 'left')
pltools.hide_ticks()

### Passive membrane parameters subplots

# Leak conductance
ax = plt.subplot2grid(grid_dims, (2, 0))
plt.title('D1 Leak conductance', loc = 'left')
plt.hist(1e3/params_5HT['R'], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$g_l$ (pS)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(1e3/params_5HT['R'])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Capacitance
ax = plt.subplot2grid(grid_dims, (2, 1))
plt.title('D2 Capacitance', loc = 'left')
plt.hist(params_5HT['C'], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$C_m$ (pF)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['C'])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Membrane time constant
ax = plt.subplot2grid(grid_dims, (2, 2))
plt.title('D3 Time constant', loc = 'left')
plt.hist(params_5HT['R'] * params_5HT['C'] * 1e-3, color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\\tau_m$ (ms)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['R'] * params_5HT['C'] * 1e-3)
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Estimated resting membrane potential
ax = plt.subplot2grid(grid_dims, (2, 3))
plt.title('D4 Equilibrium potential', loc = 'left')
plt.hist(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])], color = hist_color)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\hat{{E}}_l$ (mV)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pltools.p_to_string(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

#plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

plt.show()
