#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


#%% READ IN DATA

params = pd.read_csv('data/DRN_membrane_parameters.csv')

params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)
params_non5HT = params.loc[np.where(params['TdT'] == 0)]
params_non5HT.drop('TdT', axis = 1, inplace = True)


#%% PREPARE FIGURE FOR 5HT CELLS

plt.figure(figsize = (10, 8))

plt.suptitle('Passive membrane parameters of ~70 5HT neurons')

ax = plt.subplot(221)
plt.title('A. Leak conductance', loc = 'left')
plt.hist(1e3/params_5HT['R'], color = (0.9, 0.2, 0.2))
plt.ylabel('No. cells')
plt.xlabel('$g_l$ (pS)')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.text(0.98, 0.98,
'Shapiro-Wilk $W = {:.3f}$, $p = {:.3f}$'.format(stats.shapiro(1e3/params_5HT['R'])[0], stats.shapiro(1e3/params_5HT['R'])[1]),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

ax = plt.subplot(222)
plt.title('B. Capacitance', loc = 'left')
plt.hist(params_5HT['C'], color = (0.9, 0.2, 0.2))
plt.ylabel('No. cells')
plt.xlabel('$C$ (pF)')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.text(0.98, 0.98,
'Shapiro-Wilk $W = {:.3f}$, $p = {:.3f}$'.format(stats.shapiro(params_5HT['C'])[0], stats.shapiro(params_5HT['C'])[1]),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

ax = plt.subplot(223)
plt.title('C. Membrane time constant', loc = 'left')
plt.hist(params_5HT['R'] * params_5HT['C'] * 1e-3, color = (0.9, 0.2, 0.2))
plt.ylabel('No. cells')
plt.xlabel('$\\tau_{{m}}$ (ms)')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.text(0.98, 0.98,
'Shapiro-Wilk $W = {:.3f}$, $p = {:.3f}$'.format(stats.shapiro(params_5HT['R'] * params_5HT['C'])[0], stats.shapiro(params_5HT['R'] * params_5HT['C'])[1]),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

ax = plt.subplot(224)
plt.title('D. Estimated reversal potential', loc = 'left')
plt.hist(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])], color = (0.9, 0.2, 0.2))
plt.ylabel('No. cells')
plt.xlabel('$\hat{{E}}_l$ (mV)')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.text(0.98, 0.98,
'Shapiro-Wilk $W = {:.3f}$, $p = {:.3f}$'.format(stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])[0], stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])[1]),
verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

plt.subplots_adjust(top = 0.9, hspace = 0.4, wspace = 0.3)

plt.savefig('/Users/eharkin/Desktop/reference5HTParams.png', dpi = 300)

plt.show()
