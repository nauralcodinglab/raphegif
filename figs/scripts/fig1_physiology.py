#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./analysis/gating/')

from cell_class import Cell, Recording


#%% DEFINE USEFUL FUNCTIONS

def hideTicks(ax = None):

    """
    Delete the x and y ticks of the specified axes. If no axes object is provided, defaults to the current axes.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])


def removeBorder(ax = None, sides = 'a'):

    """
    Sides should be set to a for all, or a string containing r/l/t/b as needed.
    """

    # Check for correct input
    if not any([letter in sides for letter in 'arltb']):
        raise ValueError('sides should be passed a string with `a` for all sides, or r/l/t/b as-needed for other sides.')

    if ax is None:
        ax = plt.gca()

    if sides == 'a':
        sides = 'rltb'

    sidekeys = {
    'r': 'right',
    'l': 'left',
    't': 'top',
    'b': 'bottom'
    }

    for key, side in sidekeys.iteritems():

        if key not in sides:
            continue
        else:
            ax.spines[side].set_visible(False)


def pValToString(p):

    """
    Takes a p-value and converts it to a pretty LaTeX string.

    p is presented to three decimal places if p >= 0.05, and as p < 0.05/0.01/0.001 otherwise.
    """

    p_rounded = np.round(p, 3)

    if p_rounded >= 0.05:
        p_str = '$p = {}$'.format(p_rounded)
    elif p_rounded < 0.05 and p_rounded >= 0.01:
        p_str = '$p < 0.05$'
    elif p_rounded < 0.01 and p_rounded >= 0.001:
        p_str = '$p < 0.01$'
    else:
        p_str = '$p < 0.001$'

    return p_str


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
hideTicks()

plt.subplot2grid(grid_dims, (0, 2))
plt.title('A2 DRN inputs', loc = 'left')
hideTicks()

plt.subplot2grid(grid_dims, (0, 3))
plt.title('A3 DRN outputs', loc = 'left')
hideTicks()

plt.subplot2grid(grid_dims, (1, 0), colspan = 2)
plt.title('B Identification of DRN 5HT neurons', loc = 'left')
hideTicks()

plt.subplot2grid(grid_dims, (1, 2))
plt.title('C1 Spiking', loc = 'left')
hideTicks()

plt.subplot2grid(grid_dims, (1, 3))
plt.title('C2 Voltage steps', loc = 'left')
hideTicks()

### Passive membrane parameters subplots

# Leak conductance
ax = plt.subplot2grid(grid_dims, (2, 0))
plt.title('D1 Leak conductance', loc = 'left')
plt.hist(1e3/params_5HT['R'], color = hist_color)
removeBorder(sides = 'rlt')
plt.yticks([])
plt.xlabel('$g_l$ (pS)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(1e3/params_5HT['R'])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pValToString(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Capacitance
ax = plt.subplot2grid(grid_dims, (2, 1))
plt.title('D2 Capacitance', loc = 'left')
plt.hist(params_5HT['C'], color = hist_color)
removeBorder(sides = 'rlt')
plt.yticks([])
plt.xlabel('$C_m$ (pF)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['C'])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pValToString(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Membrane time constant
ax = plt.subplot2grid(grid_dims, (2, 2))
plt.title('D3 Time constant', loc = 'left')
plt.hist(params_5HT['R'] * params_5HT['C'] * 1e-3, color = hist_color)
removeBorder(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\\tau_m$ (ms)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['R'] * params_5HT['C'] * 1e-3)
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pValToString(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

# Estimated resting membrane potential
ax = plt.subplot2grid(grid_dims, (2, 3))
plt.title('D4 Equilibrium potential', loc = 'left')
plt.hist(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])], color = hist_color)
removeBorder(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\hat{{E}}_l$ (mV)')
plt.ylim(0, plt.ylim()[1] * 1.1)
shapiro_w, shapiro_p = stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])
plt.text(0.98, 0.98,
'Shapiro normality test {}'.format(pValToString(shapiro_p)),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

#plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

plt.show()
