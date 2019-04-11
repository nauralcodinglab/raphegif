#%% IMPORT MODULES

from __future__ import division
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

sys.path.append('analysis/subthresh_mod_selection')
from ModMats import ModMats
from src.SubthreshGIF_K import SubthreshGIF_K


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


#%% CRAPPY BARPLOT OF R^2 ON dV

plt.figure()

mods = [ohmic_mod_coeffs, gk1_mod_coeffs, gk2_mod_coeffs, full_mod_coeffs]
y = [np.mean(x['var_explained_dV']) for x in mods]
yerr = [np.std(x['var_explained_dV'])/len(x) for x in mods]

plt.errorbar([i for i in range(len(y))], y, yerr)
plt.bar([i for i in range(len(y))], y)
plt.xticks([i for i in range(len(y))], ['Ohmic', 'gk1 only', 'gk2 only', 'gk1 + gk2'])

plt.show()


#%% CRAPPY BARPLOT OF IMPROVEMENT IN R^2 ON dV

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


#%% NICE PLOT OF IMPROVEMENT IN R^2 ON dV

"""
Pretty plot of improvement in var. explained on dV (train)
"""

plt.figure(figsize = (7, 5))

exemplars = [1, 4, 11]

plt.subplot2grid((2, 3), (0, 0), colspan = 3)
plt.title('Improved fit on training set $\\frac{dV}{dt}$')

for i in range(len(gk1_change)):

    if i in exemplars:
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        '-', zorder = len(gk1_change), color = (0.9, 0.1, 0.1), alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        'o', zorder = len(gk1_change), markeredgecolor = (0.9, 0.1, 0.1), markerfacecolor = (0.9, 0.5, 0.5),
         markersize = 15, alpha = 0.8)
        plt.text(2.1, 100 * full_change[i], '{}'.format(i), verticalalignment = 'center')
    else:
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        '-', color = 'gray', alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_change[i], gk2_change[i], full_change[i]]) * 100,
        'ko', markerfacecolor = 'none', markersize = 15)
    #plt.text(-0.2, 100 * gk1_change[i], '{}'.format(i), verticalalignment = 'center')

plt.xlim(-0.2, 2.2)
plt.ylim(-0.6, 6.3)
plt.xticks([0, 1, 2], ['$g_{k1}$ only', '$g_{k2}$ only', '$g_{k1} + g_{k2}$'])
plt.ylabel('$\Delta R^2$ from linear model (%)')


plt.subplot2grid((2, 3), (1, 0), colspan = 2)

plt.title('Conductance coefficients')

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



plt.xlim(-0.3, 2.3)
plt.ylim(-14, 32)
plt.xticks([0, 1, 2], ['$g_l$', '$g_{k1}$', '$g_{k2}$'])
plt.ylabel('Conductance (pS)')


plt.subplot2grid((2, 3), (1, 2))

plt.title('Passive properties')

for i in range(len(full_mod_coeffs['R'])):

    if i in exemplars:

        plt.plot(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i],
        full_mod_coeffs['C'][i] * 1e3,
        'o', zorder = len(full_mod_coeffs['R']), markeredgecolor = (0.9, 0.1, 0.1),
        markerfacecolor = (0.9, 0.5, 0.5), alpha = 0.8)
        plt.text(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i] * 1.2,
        full_mod_coeffs['C'][i] * 1e3, '{}'.format(i), verticalalignment = 'center',
        zorder = len(full_mod_coeffs['R']) + 1)

    else:
        plt.plot(full_mod_coeffs['R'][i] * full_mod_coeffs['C'][i],
        full_mod_coeffs['C'][i] * 1e3,
        'ko', markerfacecolor = 'none')

plt.xlim(0, plt.xlim()[1])
plt.ylim(40, plt.ylim()[1] * 1.1)
plt.ylabel('C (pF)')
plt.xlabel('$\\tau_m$ (ms)')

plt.tight_layout()

#plt.savefig('/Users/eharkin/Desktop/improvedFitTraining.png', dpi = 300)

plt.show()


#%%

no = 1

print 'gk1:{:>8.3f}'.format(full_mod_coeffs['gbar_K1'][no])
print 'gk2:{:>8.3f}'.format(full_mod_coeffs['gbar_K2'][no])
print 'R:{:>10.2f}'.format(full_mod_coeffs['R'][no])
print 'C:{:>10.2f}'.format(full_mod_coeffs['C'][no])
print 'El:{:>9.2f}'.format(full_mod_coeffs['El'][no])


#%% GET MODEL FIT ON TEST DATA

bins = np.arange(-122.5, -26, 5)
bin_centres = (bins[1:] + bins[:-1])/2

KGIF = SubthreshGIF_K(0.1)

KGIF.C = 0.100 # pF
KGIF.gl = 0.001 # nS
KGIF.gbar_K1 = 0.
KGIF.gbar_K2 = 0.

KGIF.m_Vhalf = -27
KGIF.m_k = 0.113
KGIF.m_tau = 1.

KGIF.h_Vhalf = -59.9
KGIF.h_k = -0.166
KGIF.h_tau = 50.

KGIF.n_Vhalf = -16.9
KGIF.n_k = 0.114
KGIF.n_tau = 100.

KGIF.E_K = -101.

print 'GETTING PERFORMANCE ON TEST SET\nWorking',

for mod in [ohmic_mod_coeffs, gk1_mod_coeffs, gk2_mod_coeffs, full_mod_coeffs]:

    print '.',

    mod['var_explained_Vtest'] = []
    mod['binned_e2_values'] = []
    mod['binned_e2_centres'] = []

    for i in range(len(model_matrices)):

        KGIF.El = mod['El'][i]
        KGIF.C = mod['C'][i]
        KGIF.gl = 1/mod['R'][i]
        KGIF.gbar_K1 = mod.get('gbar_K1', np.zeros_like(mod['El']))[i]
        KGIF.gbar_K2 = mod.get('gbar_K2', np.zeros_like(mod['El']))[i]

        V_real = model_matrices[i].V_test
        V_sim = np.empty_like(V_real)

        for sw_ind in range(V_real.shape[1]):

            V_sim[:, sw_ind] = KGIF.simulate(
            model_matrices[i].I_test[:, sw_ind],
            V_real[0, sw_ind]
            )[1]

        mod['binned_e2_values'].append(sp.stats.binned_statistic(
        V_real.flatten(), ((V_real - V_sim)**2).flatten(), bins = bins
        )[0])
        mod['binned_e2_centres'].append(bin_centres)

        for sw_ind in range(V_real.shape[1]):
            below_V_cutoff = np.where(V_real[:, sw_ind] < model_matrices[i].VCutoff)[0]
            V_real[below_V_cutoff, sw_ind] = np.nan
            V_sim[below_V_cutoff, sw_ind] = np.nan

        var_explained_Vtest_tmp = (np.nanvar(V_real) - np.nanmean((V_real - V_sim)**2)) / np.nanvar(V_real)
        mod['var_explained_Vtest'].append(var_explained_Vtest_tmp)

    mod['binned_e2_values'] = np.array(mod['binned_e2_values']).T
    mod['binned_e2_centres'] = np.array(mod['binned_e2_centres']).T

print '\nDone!'


#%% PRETTY PLOT OF R2 V_TEST

"""
Pretty plot of improvement in var. explained on V_test
"""

gk1_test_change = [gk1 - ohmic for gk1, ohmic in zip(gk1_mod_coeffs['var_explained_Vtest'], ohmic_mod_coeffs['var_explained_Vtest'])]
gk2_test_change = [gk2 - ohmic for gk2, ohmic in zip(gk2_mod_coeffs['var_explained_Vtest'], ohmic_mod_coeffs['var_explained_Vtest'])]
full_test_change = [full - ohmic for full, ohmic in zip(full_mod_coeffs['var_explained_Vtest'], ohmic_mod_coeffs['var_explained_Vtest'])]

plt.figure(figsize = (7, 5))

exemplars = [1, 4, 11]

plt.subplot(111)
plt.title('Change in fit on test set $V$')

for i in range(len(gk1_test_change)):

    if i in exemplars:
        plt.plot([0, 1, 2], np.array([gk1_test_change[i], gk2_test_change[i], full_test_change[i]]) * 100,
        '-', zorder = len(gk1_test_change), color = (0.9, 0.1, 0.1), alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_test_change[i], gk2_test_change[i], full_test_change[i]]) * 100,
        'o', zorder = len(gk1_test_change), markeredgecolor = (0.9, 0.1, 0.1), markerfacecolor = (0.9, 0.5, 0.5),
         markersize = 15, alpha = 0.8)
        plt.text(2.1, 100 * full_test_change[i], '{}'.format(i), verticalalignment = 'center')
    else:
        plt.plot([0, 1, 2], np.array([gk1_test_change[i], gk2_test_change[i], full_test_change[i]]) * 100,
        '-', color = 'gray', alpha = 0.5)
        plt.plot([0, 1, 2], np.array([gk1_test_change[i], gk2_test_change[i], full_test_change[i]]) * 100,
        'ko', markerfacecolor = 'none', markersize = 15)

plt.xlim(-0.2, 2.2)
plt.xticks([0, 1, 2], ['$g_{k1}$ only', '$g_{k2}$ only', '$g_{k1} + g_{k2}$'])
plt.ylabel('$\Delta R^2$ from linear model (%)')
plt.xlabel('Nonlinearities')

plt.tight_layout()

plt.savefig('/Users/eharkin/Desktop/improvedFitTestBad.png', dpi = 300)

plt.show()


#%% COMPARE EACH NONLINEAR MODEL TO OHMIC MODEL

"""
The ohmic model is considered as the base model. This block adds gk1 and gk2 to
the model one at a time and together. It gets the performance of each model on
test set voltage and makes a pretty plot comparing each augmented model to the
base model across voltage bins.
"""

plt.figure(figsize = (13, 5))

for i in range(3):

    mod = [gk1_mod_coeffs, gk2_mod_coeffs, full_mod_coeffs][i]
    title_str = ['Effect of $g_{{k1}}$', 'Effect of $g_{{k2}}$', 'Effect of $g_{{k1}} + g_{{k2}}$'][i]
    mod_str = ['Linear model + $g_{{k1}}$', 'Linear model + $g_{{k2}}$', 'Linear model + $g_{{k1}}$ & $g_{{k2}}$'][i]

    plt.subplot(1, 3, i + 1)
    plt.title(title_str)
    plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
    plt.plot(ohmic_mod_coeffs['binned_e2_centres'], ohmic_mod_coeffs['binned_e2_values'],
    '-', color = (0.1, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmean(ohmic_mod_coeffs['binned_e2_centres'], axis = 1),
    np.nanmean(ohmic_mod_coeffs['binned_e2_values'], axis = 1),
    '-', color = (0.1, 0.1, 0.1), label = 'Linear model')
    plt.plot(mod['binned_e2_centres'], mod['binned_e2_values'],
    '-', color = (0.9, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmean(mod['binned_e2_centres'], axis = 1),
    np.nanmean(mod['binned_e2_values'], axis = 1),
    '-', color = (0.9, 0.1, 0.1), label = mod_str)


    for i in range(mod['binned_e2_values'].shape[0]):

        if np.isnan(np.nanmean(mod['binned_e2_values'][i, :])):
            continue

        W, p = sp.stats.wilcoxon(ohmic_mod_coeffs['binned_e2_values'][i, :],
        mod['binned_e2_values'][i, :])

        if p > 0.05 and p <= 0.1:
            p_str = 'o'
        elif p > 0.01 and p <= 0.05:
            p_str = '*'
        elif p <= 0.01:
            p_str = '**'
        else:
            p_str = ''

        plt.text(mod['binned_e2_centres'][i, 0], -30, p_str,
        horizontalalignment = 'center')


    plt.ylim(-40, 310)

    plt.legend()

    plt.xlabel('$V_m$ (mV)')
    plt.ylabel('MSE ($\mathrm{{mV}}^2$)')

plt.tight_layout()
plt.savefig('/Users/eharkin/Desktop/multimodResidualComparison.png', dpi = 300)
plt.show()


#%% ASSESS EFFECT OF ADDING GK1 TO MODEL WITH ONLY GK2

"""
Seems like gk1 doesn't do much on its own, and that gk2 is sufficient to get a
big improvement over the ohmic model. This block checks whether adding gk1 to
the gk2 model produces any improvement. (Surprisingly, it does.)
"""

plt.figure(figsize = (7, 5))

plt.subplot2grid((2, 3), (0, 0), colspan = 2, rowspan = 2)
plt.title('Effect of adding $g_{{k1}}$ given $g_{{k2}}$')
plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
plt.plot(gk2_mod_coeffs['binned_e2_centres'], gk2_mod_coeffs['binned_e2_values'],
'-', color = (0.1, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
plt.plot(np.nanmean(gk2_mod_coeffs['binned_e2_centres'], axis = 1),
np.nanmean(gk2_mod_coeffs['binned_e2_values'], axis = 1),
'-', color = (0.1, 0.1, 0.1), label = 'Linear model + $g_{{k2}}$')
plt.plot(full_mod_coeffs['binned_e2_centres'], full_mod_coeffs['binned_e2_values'],
'-', color = (0.9, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
plt.plot(np.nanmean(full_mod_coeffs['binned_e2_centres'], axis = 1),
np.nanmean(full_mod_coeffs['binned_e2_values'], axis = 1),
'-', color = (0.9, 0.1, 0.1), label = 'Linear model + $g_{{k1}}$ & $g_{{k2}}$')


for i in range(mod['binned_e2_values'].shape[0]):

    if np.isnan(np.nanmean(full_mod_coeffs['binned_e2_values'][i, :])):
        continue

    W, p = sp.stats.wilcoxon(gk2_mod_coeffs['binned_e2_values'][i, :],
    full_mod_coeffs['binned_e2_values'][i, :])

    if p > 0.05 and p <= 0.1:
        p_str = 'o'
    elif p > 0.01 and p <= 0.05:
        p_str = '*'
    elif p <= 0.01:
        p_str = '**'
    else:
        p_str = ''

    plt.text(full_mod_coeffs['binned_e2_centres'][i, 0], -30, p_str,
    horizontalalignment = 'center')

plt.text(-60, 90, 'i', horizontalalignment = 'center')
plt.text(-45, 80, 'ii', horizontalalignment = 'center')

plt.ylim(-40, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')


plt.subplot2grid((2, 3), (0, 2))
plt.title('i. Fit at -60mV')

y = np.concatenate((gk2_mod_coeffs['binned_e2_values'][-7, :][np.newaxis, :],
full_mod_coeffs['binned_e2_values'][-7, :][np.newaxis, :]),
axis = 0)
x = np.zeros_like(y)
x[1, :] = 1

plt.plot(x, y, '-', color = 'gray', alpha = 0.5)
plt.plot(x[0, :], y[0, :],
'ko', markerfacecolor = (0.5, 0.5, 0.5), markersize = 10)
plt.plot(x[1, :], y[1, :],
'o', markeredgecolor = (0.9, 0.1, 0.1), markerfacecolor = (0.9, 0.5, 0.5),
markersize = 10)
plt.text(0.5, plt.ylim()[1] * 1.05, 'n.s.',
horizontalalignment = 'center', verticalalignment = 'center')
plt.xlim(-0.2, 1.2)
plt.ylim(-6, plt.ylim()[1] * 1.2)
plt.xticks([0, 1], ['$g_{{k2}}$', '$g_{{k1}} + g_{{k2}}$'])
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')


plt.subplot2grid((2, 3), (1, 2))
plt.title('ii. Fit at -45mV')

y = np.concatenate((gk2_mod_coeffs['binned_e2_values'][-4, :][np.newaxis, :],
full_mod_coeffs['binned_e2_values'][-4, :][np.newaxis, :]),
axis = 0)
x = np.zeros_like(y)
x[1, :] = 1

plt.plot(x, y, '-', color = 'gray', alpha = 0.5)
plt.plot(x[0, :], y[0, :],
'ko', markerfacecolor = (0.5, 0.5, 0.5), markersize = 10)
plt.plot(x[1, :], y[1, :],
'o', markeredgecolor = (0.9, 0.1, 0.1), markerfacecolor = (0.9, 0.5, 0.5),
markersize = 10)
plt.text(0.5, 70, '*',
horizontalalignment = 'center', verticalalignment = 'center')
plt.xlim(-0.2, 1.2)
plt.ylim(-12, 85)
plt.xticks([0, 1], ['$g_{{k2}}$', '$g_{{k1}} + g_{{k2}}$'])
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')


plt.tight_layout()
plt.savefig('/Users/eharkin/Desktop/gk1gk2ResidualComparison.png', dpi = 300)
plt.show()


#%% TEST WHETHER NONLINEARITIES REMOVE VOLTAGE-DEPENDENCE OF MSE

"""
So far, I've shown qualitatively that the linear model systematically deviates
from the behaviour of real cells near threshold, and that adding one or two
nonlinearities fixes this. However, it would be nice to provide more systematic/
quantitative support for this idea.

This block uses the Friedman chi-square test to check whether binned MSE depends
on V in various models. The Friedman test is a non-parametric analog to a
one-way repeated measures ANOVA. Here, measurements of MSE over voltage (the
treatment variable) are repeated within cells (the blocking variable).
"""

fried_base = [ohmic_mod_coeffs['binned_e2_values'][i, :] for i in range(-10, -3)]
fried_gk1 = [gk1_mod_coeffs['binned_e2_values'][i, :] for i in range(-10, -3)]
fried_gk2 = [gk2_mod_coeffs['binned_e2_values'][i, :] for i in range(-10, -3)]
fried_full = [full_mod_coeffs['binned_e2_values'][i, :] for i in range(-10, -3)]

print 'Does MSE depend on V in various models?'
print '(Friedman test on repeated measurements of MSE across -75mV to -45mV.)\n'
print '{:>10}{:>20}{:>20}'.format('Model', 'Q-statistic', 'p-value')
print '______________________________________________________________________'
print '{:>10}{:>20.1f}{:>20.4f}'.format('Base', *sp.stats.friedmanchisquare(*fried_base))
print '{:>10}{:>20.1f}{:>20.4f}'.format('gk1', *sp.stats.friedmanchisquare(*fried_gk1))
print '{:>10}{:>20.1f}{:>20.4f}'.format('gk2', *sp.stats.friedmanchisquare(*fried_gk2))
print '{:>10}{:>20.1f}{:>20.4f}'.format('Full', *sp.stats.friedmanchisquare(*fried_full))
