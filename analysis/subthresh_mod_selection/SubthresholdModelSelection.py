#%% IMPORT MODULES

from __future__ import division
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('analysis/subthresh_mod_selection')
from ModMats import ModMats


#%% LOAD DATA

PICKLE_PATH = 'data/subthreshold_expts/compensated_recs/'

print 'LOADING DATA'
model_matrices = []

fnames = [fname for fname in os.listdir(PICKLE_PATH) if fname[-4:].lower() == '.pyc']
for fname in fnames:

    with open(PICKLE_PATH + fname, 'rb') as f:

        modmat_tmp = ModMats(0.1)
        modmat_tmp = pickle.load(f)

    model_matrices.append(modmat_tmp)

print 'Done!'


#%% FIT MODELS

ohmic_mod_coeffs = {
'El': [],
'R': [],
'C': [],
'var_explained_dV': []
}
gk1_mod_coeffs = {
'El': [],
'R': [],
'C': [],
'gbar_K1': [],
'var_explained_dV': []
}
gk2_mod_coeffs = {
'El': [],
'R': [],
'C': [],
'gbar_K2': [],
'var_explained_dV': []
}
full_mod_coeffs = {
'El': [],
'R': [],
'C': [],
'gbar_K1': [],
'gbar_K2': [],
'var_explained_dV': []
}

print 'FITTING MODELS'

for i in range(len(model_matrices)):

    print '\rFitting models to data from cell {}...'.format(i),

    mod = model_matrices[i]

    mod.setVCutoff(-80)

    ohmic_tmp = mod.fitOhmicMod()
    gk1_tmp = mod.fitGK1Mod()
    gk2_tmp = mod.fitGK2Mod()
    full_tmp = mod.fitFullMod()

    for key in ohmic_mod_coeffs.keys():
        ohmic_mod_coeffs[key].append(ohmic_tmp[key])

    for key in gk1_mod_coeffs.keys():
        gk1_mod_coeffs[key].append(gk1_tmp[key])

    for key in gk2_mod_coeffs.keys():
        gk2_mod_coeffs[key].append(gk2_tmp[key])

    for key in full_mod_coeffs.keys():
        full_mod_coeffs[key].append(full_tmp[key])

print 'Done!'


#%%

plt.figure()

mods = [ohmic_mod_coeffs, gk1_mod_coeffs, gk2_mod_coeffs, full_mod_coeffs]
y = [np.mean(x['var_explained_dV']) for x in mods]
yerr = [np.std(x['var_explained_dV']) for x in mods]

plt.errorbar([i for i in range(len(y))], y, yerr)
plt.bar([i for i in range(len(y))], y)
plt.xticks([i for i in range(len(y))], ['Ohmic', 'gk1 only', 'gk2 only', 'gk1 + gk2'])

plt.show()


#%%
gk1_change = [gk1 - ohmic for gk1, ohmic in zip(gk1_mod_coeffs['var_explained_dV'], ohmic_mod_coeffs['var_explained_dV'])]
gk2_change = [gk2 - ohmic for gk2, ohmic in zip(gk2_mod_coeffs['var_explained_dV'], ohmic_mod_coeffs['var_explained_dV'])]
full_change = [full - ohmic for full, ohmic in zip(full_mod_coeffs['var_explained_dV'], ohmic_mod_coeffs['var_explained_dV'])]

plt.figure()

y = [100 * np.mean(x) for x in [gk1_change, gk2_change, full_change]]
yerr = [100 * np.std(x)/len(x) for x in [gk1_change, gk2_change, full_change]]

plt.errorbar([i for i in range(len(y))], y, yerr, fmt = 'k')
plt.bar([i for i in range(len(y))], y)
plt.xticks([i for i in range(len(y))], ['gk1', 'gk2', 'gk1 + gk2'])

plt.show()

#%%

plt.figure()

exemplars = [1, 4, 11]

plt.subplot2grid((2, 3), (0, 0), colspan = 3)
plt.title('Improved fit on $\\frac{dV}{dt}$')

for i in range(len(gk1_change)):

    if i in exemplars:
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        '-', zorder = len(gk1_change), color = (0.9, 0.1, 0.1), alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        'o', zorder = len(gk1_change), markeredgecolor = (0.9, 0.1, 0.1), markerfacecolor = (0.9, 0.5, 0.5),
         markersize = 15, alpha = 0.8)
        plt.text(2.15, 100 * full_change[i], '{}'.format(i), verticalalignment = 'center')
    else:
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        '-', color = 'gray', alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        'ko', markerfacecolor = 'none', markersize = 15)
    #plt.text(-0.2, 100 * gk1_change[i], '{}'.format(i), verticalalignment = 'center')

plt.xlim(-0.5, 2.5)
plt.xticks([0, 1, 2], ['$g_{k1}$', '$g_{k2}$', '$g_{k1} + g_{k2}$'])
plt.ylabel('$\Delta R^2$ from linear model (%)')


plt.subplot2grid((2, 3), (1, 0), colspan = 2)

for i in range(len(full_mod_coeffs['R'])):

    if i in exemplars:
        plt.plot([0, 1, 2],
        1e3 * np.array([1/full_mod_coeffs['R'][i], full_mod_coeffs['gbar_K1'][i], full_mod_coeffs['gbar_K2'][i]]),
        '-', zorder = len(full_mod_coeffs['R']), color = (0.9, 0.1, 0.1), alpha = 0.5)
        plt.plot([0, 1, 2],
        1e3 * np.array([1/full_mod_coeffs['R'][i], full_mod_coeffs['gbar_K1'][i], full_mod_coeffs['gbar_K2'][i]]),
        'o', zorder = len(full_mod_coeffs['R']), markeredgecolor = (0.9, 0.1, 0.1),
        markerfacecolor = (0.9, 0.5, 0.5), markersize = 15, alpha = 0.8)
        plt.text(2.15, 1e3 * full_mod_coeffs['gbar_K2'][i], '{}'.format(i), verticalalignment = 'center')
    else:
        plt.plot([0, 1, 2],
        1e3 * np.array([1/full_mod_coeffs['R'][i], full_mod_coeffs['gbar_K1'][i], full_mod_coeffs['gbar_K2'][i]]),
        '-', color = 'gray', alpha = 0.5)
        plt.plot([0, 1, 2],
        1e3 * np.array([1/full_mod_coeffs['R'][i], full_mod_coeffs['gbar_K1'][i], full_mod_coeffs['gbar_K2'][i]]),
        'ko', markerfacecolor = 'none', markersize = 15)



plt.xlim(-0.5, 2.5)
plt.xticks([0, 1, 2], ['$g_l$', '$g_{k1}$', '$g_{k2}$'])
plt.ylabel('Conductance (pS)')


plt.subplot2grid((2, 3), (1, 2))

for i in range(len(full_mod_coeffs['R'])):

    if i in exemplars:

        plt.plot(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i] * 1e-3,
        full_mod_coeffs['C'][i],
        'o', zorder = len(full_mod_coeffs['R']), markeredgecolor = (0.9, 0.1, 0.1),
        markerfacecolor = (0.9, 0.5, 0.5), alpha = 0.8)
        plt.text(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i] * 1e-3 + 50,
        full_mod_coeffs['C'][i], '{}'.format(i))

    else:
        plt.plot(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i] * 1e-3,
        full_mod_coeffs['C'][i],
        'ko', markerfacecolor = 'none')

plt.xlim(0, plt.xlim()[1])
plt.ylabel('C (something F)')
plt.xlabel('$\\tau$ (ms)')

plt.tight_layout()

plt.show()


#%%

no = 1

print 'gk1:{:>8.3f}'.format(full_mod_coeffs['gbar_K1'][no])
print 'gk2:{:>8.3f}'.format(full_mod_coeffs['gbar_K2'][no])
print 'R:{:>10.2f}'.format(full_mod_coeffs['R'][no])
print 'C:{:>10.2f}'.format(full_mod_coeffs['C'][no])
print 'El:{:>9.2f}'.format(full_mod_coeffs['El'][no])
