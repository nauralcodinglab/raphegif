#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./src/')
sys.path.append('./figs/scripts/')
sys.path.append('analysis/subthresh_mod_selection')

import pltools
from ModMats import ModMats
from Experiment import Experiment
from SubthreshGIF_K import SubthreshGIF_K
from AEC_Badel import AEC_Badel


#%% DEFINE FUNCTIONS TO GAG VERBOSE POZZORINI METHODS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout



#%% LOAD DATA

path = ('data/subthreshold_expts/')

cells = [['c0_AEC_18125000.abf', 'c0_Train_18125001.abf', 'c0_Test_18125002.abf'],
         ['c1_AEC_18125011.abf', 'c1_Train_18125012.abf', 'c1_Test_18125013.abf'],
         ['c2_AEC_18125026.abf', 'c2_Train_18125027.abf', 'c2_Test_18125028.abf'],
         ['c3_AEC_18126000.abf', 'c3_Train_18126001.abf', 'c3_Test_18126002.abf'],
         ['c4_AEC_18126009.abf', 'c4_Train_18126010.abf', 'c4_Test_18126011.abf'],
         ['c5_AEC_18126014.abf', 'c5_Train_18126015.abf', 'c5_Test_18126016.abf'],
         ['c6_AEC_18126020.abf', 'c6_Train_18126021.abf', 'c6_Test_18126022.abf'],
         ['c7_AEC_18126025.abf', 'c7_Train_18126026.abf', 'c7_Test_18126027.abf'],
         ['c8_AEC_18201000.abf', 'c8_Train_18201001.abf', 'c8_Test_18201002.abf'],
         ['c9_AEC_18201013.abf', 'c9_Train_18201014.abf', 'c9_Test_18201015.abf'],
         ['c10_AEC_18201030.abf', 'c10_Train_18201031.abf', 'c10_Test_18201032.abf'],
         ['c11_AEC_18201035.abf', 'c11_Train_18201036.abf', 'c11_Test_18201037.abf'],
         ['c12_AEC_18309019.abf', 'c12_Train_18309020.abf', 'c12_Test_18309021.abf'],
         ['c13_AEC_18309022.abf', 'c13_Train_18309023.abf', 'c13_Test_18309024.abf']]

experiments = []

print 'LOADING DATA'
for i in range(len(cells)):

    print '\rLoading cell {}'.format(i),

    with gagProcess():

        #Initialize experiment.
        experiment_tmp = Experiment('Cell {}'.format(i), 0.1)

        # Read in file.
        experiment_tmp.setAECTrace('Axon', fname = path + cells[i][0],
                                   V_channel = 0, I_channel = 1)
        experiment_tmp.addTrainingSetTrace('Axon', fname = path + cells[i][1],
                                           V_channel = 0, I_channel = 1)
        experiment_tmp.addTestSetTrace('Axon', fname = path + cells[i][2],
                                       V_channel = 0, I_channel = 1)

    # Store experiment in experiments list.
    experiments.append(experiment_tmp)

print '\nDone!\n'


#%% LOWPASS FILTER V AND I DATA

butter_filter_cutoff = 1000.
butter_filter_order = 3

print 'FILTERING TRACES & SETTING ROI'
for i in range(len(experiments)):

    print '\rFiltering and selecting for cell {}'.format(i),

    # Filter training data.
    for tr in experiments[i].trainingset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 59000]])


    # Filter test data.
    for tr in experiments[i].testset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 9000]])


print '\nDone!\n'


#%% PERFORM AEC

print 'PERFORMING AEC'
for i in range(len(experiments)):

    print '\rCompensating recordings from cell {}'.format(i),

    with gagProcess():

        # Initialize AEC.
        AEC_tmp = AEC_Badel(experiments[i].dt)

        # Define metaparameters.
        AEC_tmp.K_opt.setMetaParameters(length = 500,
                                        binsize_lb = experiments[i].dt,
                                        binsize_ub = 100.,
                                        slope = 5.0,
                                        clamp_period = 0.1)
        AEC_tmp.p_expFitRange = [1., 500.]
        AEC_tmp.p_nbRep = 30

        # Perform AEC.
        experiments[i].setAEC(AEC_tmp)
        experiments[i].performAEC()


print '\nDone!\n'


#%% INITIALIZE KGIF MODEL

FIGDATA_PATH = './figs/figdata/'
with open(FIGDATA_PATH + 'gating_params.pyc', 'rb') as f:

    gating_params = pickle.load(f)

KGIF = SubthreshGIF_K(0.1)

KGIF.m_Vhalf = gating_params.loc['V_half', 'm']
KGIF.m_k = gating_params.loc['k', 'm']
KGIF.m_tau = 1.

KGIF.h_Vhalf = gating_params.loc['V_half', 'h']
KGIF.h_k = gating_params.loc['k', 'h']
KGIF.h_tau = 50.

KGIF.n_Vhalf = gating_params.loc['V_half', 'n']
KGIF.n_k = gating_params.loc['k', 'n']
KGIF.n_tau = 100.

KGIF.E_K = -101.


#%% FINAGLE DATA

"""
Finagle data into a friendlier format.
"""

model_matrices = []
for experiment in experiments:

    modmat_tmp = ModMats(0.1)
    modmat_tmp.scrapeTraces(experiment)
    modmat_tmp.computeTrainingGating(KGIF)

    model_matrices.append(modmat_tmp)

del modmat_tmp


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


#%% GET MODEL FIT ON TEST DATA

bins = np.arange(-122.5, -26, 5)
bin_centres = (bins[1:] + bins[:-1])/2

KGIF = SubthreshGIF_K(0.1)

KGIF.m_Vhalf = gating_params.loc['V_half', 'm']
KGIF.m_k = gating_params.loc['k', 'm']
KGIF.m_tau = 1.

KGIF.h_Vhalf = gating_params.loc['V_half', 'h']
KGIF.h_k = gating_params.loc['k', 'h']
KGIF.h_tau = 50.

KGIF.n_Vhalf = gating_params.loc['V_half', 'n']
KGIF.n_k = gating_params.loc['k', 'n']
KGIF.n_tau = 100.

KGIF.E_K = -101.

print 'GETTING PERFORMANCE ON TEST SET\nWorking',

for i, mod in enumerate([ohmic_mod_coeffs, gk1_mod_coeffs, gk2_mod_coeffs, full_mod_coeffs]):

    print '.',

    mod['var_explained_Vtest'] = []
    mod['binned_e2_values'] = []
    mod['binned_e2_centres'] = []
    mod['simulated_testset_traces'] = []
    mod['real_testset_traces'] = []

    for j in range(len(model_matrices)):

        KGIF.El = mod['El'][j]
        KGIF.C = mod['C'][j]
        KGIF.gl = 1/mod['R'][j]
        KGIF.gbar_K1 = mod.get('gbar_K1', np.zeros_like(mod['El']))[j]
        KGIF.gbar_K2 = mod.get('gbar_K2', np.zeros_like(mod['El']))[j]

        V_real = model_matrices[j].V_test
        V_sim = np.empty_like(V_real)

        for sw_ind in range(V_real.shape[1]):

            V_sim[:, sw_ind] = KGIF.simulate(
            model_matrices[j].I_test[:, sw_ind],
            V_real[0, sw_ind]
            )[1]

        mod['simulated_testset_traces'].append(V_sim)
        mod['real_testset_traces'].append(V_real)

        mod['binned_e2_values'].append(stats.binned_statistic(
        V_real.flatten(), ((V_real - V_sim)**2).flatten(), bins = bins
        )[0])
        mod['binned_e2_centres'].append(bin_centres)

        """
        # Comment out so that residuals are computed for full V range.
        for sw_ind in range(V_real.shape[1]):
            below_V_cutoff = np.where(V_real[:, sw_ind] < model_matrices[j].VCutoff)[0]
            V_real[below_V_cutoff, sw_ind] = np.nan
            V_sim[below_V_cutoff, sw_ind] = np.nan

        var_explained_Vtest_tmp = (np.nanvar(V_real) - np.nanmean((V_real - V_sim)**2)) / np.nanvar(V_real)
        mod['var_explained_Vtest'].append(var_explained_Vtest_tmp)
        """

    mod['binned_e2_values'] = np.array(mod['binned_e2_values']).T
    mod['binned_e2_centres'] = np.array(mod['binned_e2_centres']).T

print '\nDone!'


#%% EXPLORE TRACES

"""
Exploratory series of plots to check out performance on test set in all cells.
This will be used to select a cell to use for sample traces.

Performance on cell #12 is really, really good.
Use cell #13 as an example of a good cell.
"""

for i in range(len(ohmic_mod_coeffs['real_testset_traces'])):
    plt.figure(figsize = (18, 5))

    plt.suptitle('{}'.format(i))

    plt.subplot(131)
    plt.xlim(40000, 80000)
    plt.plot(ohmic_mod_coeffs['real_testset_traces'][i].mean(axis = 1),
    'k-', linewidth = 0.5, label = 'Real data')
    plt.plot(ohmic_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'r-', linewidth = 0.5, label = 'Ohmic model')
    plt.plot(gk1_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'b-', linewidth = 0.5, label = 'Nonlinear model')
    plt.legend()

    plt.subplot(132)
    plt.xlim(40000, 80000)
    plt.plot(ohmic_mod_coeffs['real_testset_traces'][i].mean(axis = 1),
    'k-', linewidth = 0.5, label = 'Real data')
    plt.plot(ohmic_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'r-', linewidth = 0.5, label = 'Ohmic model')
    plt.plot(gk2_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'b-', linewidth = 0.5, label = 'Nonlinear model')
    plt.legend()

    plt.subplot(133)
    plt.xlim(40000, 80000)
    plt.plot(ohmic_mod_coeffs['real_testset_traces'][i].mean(axis = 1),
    'k-', linewidth = 0.5, label = 'Real data')
    plt.plot(ohmic_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'r-', linewidth = 0.5, label = 'Ohmic model')
    plt.plot(full_mod_coeffs['simulated_testset_traces'][i].mean(axis = 1),
    'b-', linewidth = 0.5, label = 'Nonlinear model')
    plt.legend()

    plt.show()


#%% DEFINE FUNCTIONS FOR MAKING COMPLICATED PLOTS

def binned_e_comparison_plot(null_mod, alt_mod, null_mod_label = None, alt_mod_label = None,
                             null_color = (0.1, 0.1, 0.1), alt_color = (0.8, 0.2, 0.2),
                             wilcoxon = True,
                             ax = None):

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
    plt.plot(null_mod['binned_e2_centres'], null_mod['binned_e2_values'],
    '-', color = null_color, linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmean(null_mod['binned_e2_centres'], axis = 1),
    np.nanmean(null_mod['binned_e2_values'], axis = 1),
    '-', color = null_color, label = null_mod_label)
    plt.plot(alt_mod['binned_e2_centres'], alt_mod['binned_e2_values'],
    '-', color = alt_color, linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmean(alt_mod['binned_e2_centres'], axis = 1),
    np.nanmean(alt_mod['binned_e2_values'], axis = 1),
    '-', color = alt_color, label = alt_mod_label)

    if wilcoxon:
        for i in range(mod['binned_e2_values'].shape[0]):

            if np.isnan(np.nanmean(null_mod['binned_e2_values'][i, :])):
                continue

            W, p = stats.wilcoxon(null_mod['binned_e2_values'][i, :],
            alt_mod['binned_e2_values'][i, :])

            if p > 0.05 and p <= 0.1:
                p_str = 'o'
            elif p > 0.01 and p <= 0.05:
                p_str = '*'
            elif p <= 0.01:
                p_str = '**'
            else:
                p_str = ''

            plt.text(null_mod['binned_e2_centres'][i, 0], -30, p_str,
            horizontalalignment = 'center')

#np.where((gk2_mod_coeffs['binned_e2_centres'] == -60).all(axis = 1))

def single_bin_e_comparison_plot(null_mod, alt_mod, V, null_mod_label = None, alt_mod_label = None,
                                 null_markeredgecolor = (0.1, 0.1, 0.1), null_markerfacecolor = (0.5, 0.5, 0.5),
                                 alt_markeredgecolor = (0.9, 0.1, 0.1), alt_markerfacecolor = (0.9, 0.5, 0.5),
                                 ax = None):

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    if not np.array_equal(null_mod['binned_e2_centres'], alt_mod['binned_e2_centres']):
        raise ValueError('Voltage bin centres seem to differ between null and alt models.')

    ind_of_vbin = np.where((np.abs(null_mod['binned_e2_centres'] - V) < 1e-3).all(axis = 1))[0]

    if len(ind_of_vbin) < 1:
        raise ValueError('No bins centred around approximately {}'.format(V))
    elif len(ind_of_vbin) > 1:
        raise ValueError('Multiple bins apparently centred around {}. This can happen if your bins are smaller than 10^-3 in units of the argument `V`.'.format(V))

    ind_of_vbin = ind_of_vbin[0]

    y = np.concatenate((null_mod['binned_e2_values'][ind_of_vbin, :][np.newaxis, :],
    alt_mod['binned_e2_values'][ind_of_vbin, :][np.newaxis, :]),
    axis = 0)
    x = np.zeros_like(y)
    x[1, :] = 1

    plt.plot(x, y, '-', color = 'gray', alpha = 0.5)
    plt.plot(x[0, :], y[0, :],
    'o', markerfacecolor = null_markerfacecolor, markeredgecolor = null_markeredgecolor, markersize = 10)
    plt.plot(x[1, :], y[1, :],
    'o', markerfacecolor = alt_markerfacecolor, markeredgecolor = alt_markeredgecolor, markersize = 10)
    plt.text(0.5, plt.ylim()[1] * 1.05,
    pltools.p_to_string(stats.wilcoxon(y[0, :], y[1, :])[1]),
    horizontalalignment = 'center', verticalalignment = 'center')
    plt.xlim(-0.2, 1.2)
    plt.ylim(plt.ylim()[0], plt.ylim()[1] * 1.2)
    plt.xticks([0, 1], [null_mod_label, alt_mod_label])


def testset_traces_plot(null_mod, alt_mod, cell_no = 0, null_mod_label = None, alt_mod_label = None,
                        real_color = (0, 0, 0), null_color = (0.1, 0.1, 0.9),
                        alt_color = (0.9, 0.1, 0.1), ax = None):

    """
    Compare testset traces for one cell.

    Note: averages sweeps from testset before plotting (looks nicer)

    Inputs:

        null_mod/alt_mod: `x_mod_coeffs`-style dicts

        cell: int
        --  Index of cell from null_mod/alt_mod for which to plot traces.
    """

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    plt.plot(null_mod['real_testset_traces'][cell_no].mean(axis = 1),
    '-', color = real_color, linewidth = 0.5, label = 'Real data')
    plt.plot(null_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = null_color, linewidth = 0.5, label = null_mod_label)
    plt.plot(alt_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = alt_color, linewidth = 0.5, label = alt_mod_label)
    plt.legend()


#%% MAKE SOME LATEX TEXT FOR MODEL DEFINITIONS

ohmic_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l)$'
gk1_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - \\bar{{g}}_{{k1}}mh\\times(V(t) - E_k)$'
gk2_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - \\bar{{g}}_{{k2}}n\\times(V(t) - E_k)$'
full_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - (\\bar{{g}}_{{k1}}mh + \\bar{{g}}_{{k2}}n)\\times(V(t) - E_k)$'

#%% ASSEMBLE FIGURE

plt.figure(figsize = (14.67, 18))

spec = gridspec.GridSpec(6, 5,
left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 0.4, wspace = 0.4)


plt.subplot(spec[0, :2])
plt.title('A1 Model definitions', loc = 'left')
plt.text(0.1, 0.5,
'\n'.join([ohmic_latex, gk1_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[1, :2])
plt.title('A2 Predictions on test set', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk1_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear model + $g_{{k1}}$'
)
plt.axhline(-70, color = 'k', linewidth = 0.5, linestyle = 'dashed')
plt.xlim(40000, 80000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV')


plt.subplot(spec[0:2, 2:4])
plt.title('A3 Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk1_mod_coeffs,
'Linear model', 'Linear model + $g_{{k1}}$')

plt.text(-60, 90, 'i', horizontalalignment = 'center')
plt.text(-45, 80, 'ii', horizontalalignment = 'center')

plt.ylim(-40, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[0, 4])
plt.title('A4 Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -60,
                             'Linear', 'Linear + $g_{{k1}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[1, 4])
plt.title('A5 Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -45,
                             'Linear', 'Linear + $g_{{k1}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


# B: gk2 vs ohmic
plt.subplot(spec[2, :2])
plt.title('B1 Model definitions', loc = 'left')
plt.text(0.1, 0.5,
'\n'.join([ohmic_latex, gk2_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[3, :2])
plt.title('B2 Predictions on test set', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk2_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear model + $g_{{k2}}$'
)
plt.axhline(-70, color = 'k', linewidth = 0.5, linestyle = 'dashed')
plt.xlim(40000, 80000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV')

plt.subplot(spec[2:4, 2:4])
plt.title('B3 Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk2_mod_coeffs,
'Linear model', 'Linear model + $g_{{k2}}$')

plt.text(-60, 90, 'i', horizontalalignment = 'center')
plt.text(-45, 80, 'ii', horizontalalignment = 'center')

plt.ylim(-40, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[2, 4])
plt.title('B4 Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -60,
                             'Linear', 'Linear + $g_{{k2}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[3, 4])
plt.title('B5 Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -45,
                             'Linear', 'Linear + $g_{{k2}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


# C: gk1 given gk2
plt.subplot(spec[4, :2])
plt.title('C1 Model definitions', loc = 'left')
plt.text(0.1, 0.5,
'\n'.join([gk2_latex, full_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[5, :2])
plt.title('C2 Predictions on test set', loc = 'left')
testset_traces_plot(
gk2_mod_coeffs, full_mod_coeffs, 13,
null_mod_label = 'Linear model + $g_{{k2}}$',
alt_mod_label = 'Linear model + $g_{{k1}}$ & $g_{{k2}}$'
)
plt.axhline(-70, color = 'k', linewidth = 0.5, linestyle = 'dashed')
plt.xlim(40000, 80000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV')

plt.subplot(spec[4:6, 2:4])
plt.title('C3 Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
gk2_mod_coeffs, full_mod_coeffs,
'Linear model + $g_{{k2}}$', 'Linear model + $g_{{k1}}$ & $g_{{k2}}$')

plt.text(-60, 90, 'i', horizontalalignment = 'center')
plt.text(-45, 80, 'ii', horizontalalignment = 'center')

plt.ylim(-40, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[4, 4])
plt.title('C4 Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(gk2_mod_coeffs, full_mod_coeffs, -60,
                             '$g_{{k2}}$', '$g_{{k1}} + g_{{k2}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[5, 4])
plt.title('C5 Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(gk2_mod_coeffs, full_mod_coeffs, -45,
                             '$g_{{k2}}$', '$g_{{k1}} + g_{{k2}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


plt.savefig('/Users/eharkin/Desktop/fig4poster.png', dpi = 300)
plt.show()
