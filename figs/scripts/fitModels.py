"""
Process model performance on test set for figs. 2 and 4.
"""

#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import sys
sys.path.append('./src/')
sys.path.append('analysis/subthresh_mod_selection')

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
    mod['real_testset_current'] = []
    mod['binned_e2_edges'] = []

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
        mod['real_testset_current'].append(model_matrices[j].I_test)

        mod['binned_e2_values'].append(stats.binned_statistic(
        V_real.flatten(), ((V_real - V_sim)**2).flatten(), bins = bins
        )[0])
        mod['binned_e2_centres'].append(bin_centres)
        mod['binned_e2_edges'].append(bins)

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


#%% PICKLE PROCESSED DATA

PICKLE_PATH = './figs/figdata/'

pickle_labels = {
'ohmic_mod.pyc': ohmic_mod_coeffs,
'gk1_mod.pyc': gk1_mod_coeffs,
'gk2_mod.pyc': gk2_mod_coeffs,
'full_mod.pyc': full_mod_coeffs
}

for label, mod in pickle_labels.iteritems():

    with open(PICKLE_PATH + label, 'wb') as f:

        pickle.dump(mod, f)
