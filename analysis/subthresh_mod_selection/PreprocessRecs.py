"""
Process raw subthreshold data so it can be used to fit models without using
Pozzorini tools.

Open ABF recordings, lowpass filter, perform AEC, fit gating vectors for KConds,
put everything in numpy arrays for ease of use and pickle.
"""

#%% IMPORT MODULES

from __future__ import division
import os
import pickle

import numpy as np

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

v_reject_thresh = -80.

print 'FILTERING TRACES & SETTING ROI'
for i in range(len(experiments)):

    print '\rFiltering and selecting for cell {}'.format(i),

    # Filter training data.
    for tr in experiments[i].trainingset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 59000]])

        boolvec = tr.V > v_reject_thresh
        boolvec[:10000] = False

        tr.setROI_Bool(boolvec)


    # Filter test data.
    for tr in experiments[i].testset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 9000]])

        boolvec = tr.V > v_reject_thresh
        boolvec[:10000] = False

        tr.setROI_Bool(boolvec)

print '\nDone!\n'


#%% PERFORM AEC

AEC_objs = []

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

    # Save AEC to AEC_objs list.
    AEC_objs.append(AEC_tmp)

print '\nDone!\n'


#%% INITIALIZE KGIF MODEL

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


#%% GET DATA TO PICKLE

"""
Finagle data into a friendlier format and pickle.
"""

# Grab data from experiments list.
PICKLE_PATH = 'data/subthreshold_expts/compensated_recs/'

print 'GRABBING EXPERIMENTAL DATA'
for i in range(len(experiments)):

    print '\rStoring data from cell {}...'.format(i),

    modmat_tmp = ModMats(0.1)
    modmat_tmp.scrapeTrainingData(experiments[i])
    modmat_tmp.computeTrainingGating(KGIF)
    modmat_tmp.pickle(PICKLE_PATH + 'c{}.pyc'.format(i))

print 'Done!'
