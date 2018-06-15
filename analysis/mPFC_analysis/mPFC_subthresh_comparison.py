#%% IMPORT MODULES

from __future__ import division
import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

# Import GIF toolbox modules from read-only clone
import sys
sys.path.append('src')
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

sert_path = ('data/subthreshold_expts/')

sert_cells = [['c0_AEC_18125000.abf', 'c0_Train_18125001.abf', 'c0_Test_18125002.abf'],
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

experiments = {
    'sert':[],
    'pyr':[]
}

print 'LOADING DATA'
### Load 5HT data
for i in range(len(sert_cells)):

    print '\rLoading cell {}'.format(i),

    with gagProcess():

        #Initialize experiment.
        experiment_tmp = Experiment('Cell {}'.format(i), 0.1)

        # Read in file.
        experiment_tmp.setAECTrace('Axon', fname = sert_path + sert_cells[i][0],
                                   V_channel = 0, I_channel = 1)
        experiment_tmp.addTrainingSetTrace('Axon', fname = sert_path + sert_cells[i][1],
                                           V_channel = 0, I_channel = 1)
        experiment_tmp.addTestSetTrace('Axon', fname = sert_path + sert_cells[i][2],
                                       V_channel = 0, I_channel = 1)

    # Store experiment in experiments list.
    experiments['sert'].append(experiment_tmp)

### Load pyramidal cell data.
FNAMES_PATH = './data/mPFC/fnames.csv'
DATA_PATH = './data/mPFC/mPFC_subthresh/'
dt = 0.1

make_plots = True

fnames = pd.read_csv(FNAMES_PATH)

for i in range(fnames.shape[0]):

    if fnames.loc[i, 'TTX'] == 1:

        tmp_experiment = Experiment(fnames.loc[i, 'Experimenter'] + fnames.loc[i, 'Cell_ID'], dt)
        try:
            tmp_experiment.setAECTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'AEC'],
                V_channel = 0, I_channel = 1)
            tmp_experiment.addTrainingSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Train'],
                V_channel = 0, I_channel = 1)
            tmp_experiment.addTestSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Test'],
                V_channel = 0, I_channel = 1)
        except IndexError:
            print('Excepted index error. Continuing.')
            continue
        except IOError:
            print('Excepted IO error. Continuing.')
            continue

        if make_plots:
            tmp_experiment.plotTrainingSet()

        experiments['pyr'].append(tmp_experiment)

    else:
        continue

print '\nDone!\n'


#%% LOWPASS FILTER V AND I DATA

butter_filter_cutoff = 1000.
butter_filter_order = 3

print 'FILTERING TRACES & SETTING ROI'
for key, experiment_ls in experiments.iteritems():
    for i in range(len(experiment_ls)):

        print '\rFiltering and selecting for cell {}'.format(i),

        # Filter training data.
        for tr in experiment_ls[i].trainingset_traces:
            tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
            tr.setROI([[1000, 59000]])


        # Filter test data.
        for tr in experiment_ls[i].testset_traces:
            tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
            tr.setROI([[1000, 9000]])


    print '\nDone!\n'


#%% PERFORM AEC

print 'PERFORMING AEC'
for key, experiment_ls in experiments.iteritems():
    for i in range(len(experiment_ls)):

        print '\rCompensating recordings from cell {}'.format(i),

        with gagProcess():

            # Initialize AEC.
            AEC_tmp = AEC_Badel(experiment_ls[i].dt)

            # Define metaparameters.
            AEC_tmp.K_opt.setMetaParameters(length = 500,
                                            binsize_lb = experiment_ls[i].dt,
                                            binsize_ub = 100.,
                                            slope = 5.0,
                                            clamp_period = 0.1)
            AEC_tmp.p_expFitRange = [1., 500.]
            AEC_tmp.p_nbRep = 30

            # Perform AEC.
            experiment_ls[i].setAEC(AEC_tmp)
            experiment_ls[i].performAEC()


print '\nDone!\n'

#%% INITIALIZE KGIF

KGIF = SubthreshGIF_K(0.1)

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


#%% PICKLE DATA

"""
Finagle data into a friendlier format and pickle.
"""

# Grab data from experiments list.
PICKLE_PATH = 'data/mPFC/mPFC_subthresh/'

print 'GRABBING EXPERIMENTAL DATA'
for key, experiment_ls in experiments.iteritems():
    for i in range(len(experiment_ls)):

        print '\rStoring data from {} cell {}...'.format(key, i),

        modmat_tmp = ModMats(0.1)
        modmat_tmp.scrapeTraces(experiment_ls[i])
        modmat_tmp.computeTrainingGating(KGIF)
        modmat_tmp.pickle(PICKLE_PATH + '{}c{}.pyc'.format(key, i))

    print '\nDone!'

#%% UNPICKLE DATA

PICKLE_PATH = 'data/mPFC/mPFC_subthresh/'

print 'LOADING DATA'
model_matrices = {
'pyr': [],
'sert': []
}

fnames = [fname for fname in os.listdir(PICKLE_PATH) if fname[-4:].lower() == '.pyc']
for fname in fnames:

    with open(PICKLE_PATH + fname, 'rb') as f:

        modmat_tmp = ModMats(0.1)
        modmat_tmp = pickle.load(f)

    if fname[:3] == 'pyr':
        model_matrices['pyr'].append(modmat_tmp)
    elif fname[:4] == 'sert':
        model_matrices['sert'].append(modmat_tmp)
    else:
        print('Prefix not understood. Skipping.')
        continue

print 'Done!'


#%% FIT MODELS

class CoeffsContainer(object):

    def __init__(self):

        template_dict = {
            'El': [],
            'R': [],
            'C': [],
            'var_explained_dV': []
        }

        self.ohmic_mod = deepcopy(template_dict)

        self.gk1_mod = deepcopy(template_dict)
        self.gk1_mod['gbar_K1'] = []

        self.gk2_mod = deepcopy(template_dict)
        self.gk2_mod['gbar_K2'] = []

        self.full_mod = deepcopy(template_dict)
        self.full_mod['gbar_K1'] = []
        self.full_mod['gbar_K2'] = []

coeff_containers = {
    'pyr': CoeffsContainer(),
    'sert': CoeffsContainer()
}


print 'FITTING MODELS'

for celltype in coeff_containers.keys():
    for i in range(len(model_matrices[celltype])):

        print '\rFitting models to data from cell {}...'.format(i),

        mod = model_matrices[celltype][i]

        mod.setVCutoff(-80)

        ohmic_tmp = mod.fitOhmicMod()
        gk1_tmp = mod.fitGK1Mod()
        gk2_tmp = mod.fitGK2Mod()
        full_tmp = mod.fitFullMod()

        for key in coeff_containers[celltype].ohmic_mod.keys():
            coeff_containers[celltype].ohmic_mod[key].append(ohmic_tmp[key])

        for key in coeff_containers[celltype].gk1_mod.keys():
            coeff_containers[celltype].gk1_mod[key].append(gk1_tmp[key])

        for key in coeff_containers[celltype].gk2_mod.keys():
            coeff_containers[celltype].gk2_mod[key].append(gk2_tmp[key])

        for key in coeff_containers[celltype].full_mod.keys():
            coeff_containers[celltype].full_mod[key].append(full_tmp[key])

    print 'Done!'


#%% GET PERFORMANCE ON TEST SET

bins = np.arange(-122.5, -26, 5)
bin_centres = (bins[1:] + bins[:-1])/2

print 'GETTING PERFORMANCE ON TEST SET\nWorking',

for celltype in coeff_containers.keys():

    co_cont = coeff_containers[celltype]
    mo_mat = model_matrices[celltype]

    for mod in [co_cont.ohmic_mod, co_cont.gk1_mod, co_cont.gk2_mod, co_cont.full_mod]:

        print '.',

        mod['var_explained_Vtest'] = []
        mod['binned_e2_values'] = []
        mod['binned_e2_centres'] = []

        for i in range(len(model_matrices[celltype])):

            KGIF.El = mod['El'][i]
            KGIF.C = mod['C'][i]
            KGIF.gl = 1/mod['R'][i]
            KGIF.gbar_K1 = mod.get('gbar_K1', np.zeros_like(mod['El']))[i]
            KGIF.gbar_K2 = mod.get('gbar_K2', np.zeros_like(mod['El']))[i]

            V_real = mo_mat[i].V_test
            V_sim = np.empty_like(V_real)

            for sw_ind in range(V_real.shape[1]):

                V_sim[:, sw_ind] = KGIF.simulate(
                mo_mat[i].I_test[:, sw_ind],
                V_real[0, sw_ind]
                )[1]

            mod['binned_e2_values'].append(sp.stats.binned_statistic(
            V_real.flatten(), ((V_real - V_sim)**2).flatten(), bins = bins
            )[0])
            mod['binned_e2_centres'].append(bin_centres)

            for sw_ind in range(V_real.shape[1]):
                below_V_cutoff = np.where(V_real[:, sw_ind] < mo_mat[i].VCutoff)[0]
                V_real[below_V_cutoff, sw_ind] = np.nan
                V_sim[below_V_cutoff, sw_ind] = np.nan

            var_explained_Vtest_tmp = (np.nanvar(V_real) - np.nanmedian((V_real - V_sim)**2)) / np.nanvar(V_real)
            mod['var_explained_Vtest'].append(var_explained_Vtest_tmp)

        mod['binned_e2_values'] = np.array(mod['binned_e2_values']).T
        mod['binned_e2_centres'] = np.array(mod['binned_e2_centres']).T

print '\nDone!'

#%%

"""
The ohmic model is considered as the base model. This block adds gk1 and gk2 to
the model one at a time and together. It gets the performance of each model on
test set voltage and makes a pretty plot comparing each augmented model to the
base model across voltage bins.
"""

plt.figure(figsize = (10, 8))
spec = plt.GridSpec(2, 3)

for celltype in coeff_containers.keys():

    co_cont = coeff_containers[celltype]
    if celltype == 'sert':
        row = 0
    elif celltype == 'pyr':
        row = 1

    for i in range(3):

        ohmic_mod_coeffs_ = co_cont.ohmic_mod

        mod = [co_cont.gk1_mod, co_cont.gk2_mod, co_cont.full_mod][i]
        title_str = ['Effect of $g_{{k1}}$', 'Effect of $g_{{k2}}$', 'Effect of $g_{{k1}} + g_{{k2}}$'][i]
        mod_str = ['Linear model + $g_{{k1}}$', 'Linear model + $g_{{k2}}$', 'Linear model + $g_{{k1}}$ & $g_{{k2}}$'][i]

        plt.subplot(spec[row, i])
        plt.title(title_str + 'in {}'.format(celltype))
        plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
        plt.plot(ohmic_mod_coeffs_['binned_e2_centres'], ohmic_mod_coeffs_['binned_e2_values'],
        '-', color = (0.1, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
        plt.plot(np.nanmedian(ohmic_mod_coeffs_['binned_e2_centres'], axis = 1),
        np.nanmedian(ohmic_mod_coeffs_['binned_e2_values'], axis = 1),
        '-', color = (0.1, 0.1, 0.1), label = 'Linear model')
        plt.plot(mod['binned_e2_centres'], mod['binned_e2_values'],
        '-', color = (0.9, 0.1, 0.1), linewidth = 0.5, alpha = 0.3)
        plt.plot(np.nanmedian(mod['binned_e2_centres'], axis = 1),
        np.nanmedian(mod['binned_e2_values'], axis = 1),
        '-', color = (0.9, 0.1, 0.1), label = mod_str)


        for i in range(mod['binned_e2_values'].shape[0]):

            if np.isnan(np.nanmedian(mod['binned_e2_values'][i, :])):
                continue

            W, p = sp.stats.wilcoxon(ohmic_mod_coeffs_['binned_e2_values'][i, :],
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

        if celltype == 'sert':
            plt.ylim(-40, 310)
        elif celltype == 'pyr':
            plt.ylim(-40, plt.ylim()[1])
        else:
            pass

        plt.legend(loc = 1)

        plt.xlabel('$V_m$ (mV)')
        plt.ylabel('MSE ($\mathrm{{mV}}^2$)')

    plt.tight_layout()

#plt.savefig('analysis/mPFC_analysis/sert_vs_pyr_subthresh.png', dpi = 300)
plt.show()
