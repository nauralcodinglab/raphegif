"""
SUBTHRESHOLD GIF ANALYSIS
"""

#%% IMPORT MODULES

from __future__ import division
import os

import numpy as np
import matplotlib.pyplot as plt

# Import GIF toolbox modules from read-only clone
import sys
sys.path.append('../src')

from Experiment import Experiment
from SubthreshGIF import SubthreshGIF
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

path = ('../data/subthreshold_expts/')

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
         ['c11_AEC_18201035.abf', 'c11_Train_18201036.abf', 'c11_Test_18201037.abf']]

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

butter_filter_cutoff = 2000.
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

AEC_objs = []

print 'PERFORMING AEC'
for i in range(len(experiments)):
    
    print '\rCompensating recordings from cell {}'.format(i),
    
    with gagProcess():
        
        # Initialize AEC.
        AEC_tmp = AEC_Badel(experiments[i].dt)
        
        # Define metaparameters.
        AEC_tmp.K_opt.setMetaParameters(length = 150.0, 
                                        binsize_lb = experiments[i].dt, 
                                        binsize_ub = 2.0, 
                                        slope = 30.0, 
                                        clamp_period = 1.0)
        AEC_tmp.p_expFitRange = [3.0, 150.0]  
        AEC_tmp.p_nbRep = 10
        
        # Perform AEC.
        experiments[i].setAEC(AEC_tmp)
        experiments[i].performAEC()

    # Save AEC to AEC_objs list.
    AEC_objs.append(AEC_tmp)
        
print '\nDone!\n'


#%% FIT GIF

Base_GIFs = []
KCond_GIFs = []


plt.figure()
dV_p = plt.subplot(121)
plt.title('Var explained on dV')
plt.ylabel('Var explained (%)')
plt.xticks([0, 1], ['Base model', 'Cond model'])

V_p = plt.subplot(122)
plt.title('Var explained on V')
plt.ylabel('Var explained (%)')
plt.xticks([0, 1], ['Base model', 'Cond model'])


print 'FITTING GIFs'
print '\nCell no.{:>10}{:>10}{:>10}{:>10}'.format('Base dV', 'Base V', 'K dV', 'K V')
print '_________________________________________________________'
for i in range(len(experiments)):
    
    with gagProcess():
        
        # Initialize GIF.
        GIF_tmp = SubthreshGIF(experiments[i].dt)
        
        # Perform fit.
        GIF_tmp.fit(experiments[i])
        
        
        # Initialize KGIF.
        KGIF_tmp = SubthreshGIF_K(experiments[i].dt)
        
        KGIF_tmp = SubthreshGIF_K(0.1)

        # Define parameters
        KGIF_tmp.m_Vhalf = -27
        KGIF_tmp.m_k = 0.113
        KGIF_tmp.m_tau = 1.
        
        KGIF_tmp.h_Vhalf = -59.9
        KGIF_tmp.h_k = -0.166
        KGIF_tmp.h_tau = 50.
        
        KGIF_tmp.n_Vhalf = -16.9
        KGIF_tmp.n_k = 0.114
        KGIF_tmp.n_tau = 100.
        
        KGIF_tmp.E_K = -101.
        
        # Fit KGIF.
        KGIF_tmp.fit(experiments[i])
        
    
    base_vexp_dV    = 100. * np.round(GIF_tmp.var_explained_dV, 3)
    base_vexp_V     = 100. * np.round(GIF_tmp.var_explained_V, 3)
    K_vexp_dV   = 100. * np.round(KGIF_tmp.var_explained_dV, 3)
    K_vexp_V    = 100. * np.round(KGIF_tmp.var_explained_V, 3)
    
    dV_p.plot([0, 1], [base_vexp_dV, K_vexp_dV], 'k-')
    V_p.plot([0, 1], [base_vexp_V, K_vexp_V], 'k-')
    
    print '{:>3}{:>10}%{:>10}%{:>10}%{:>10}%'.format(
            i,
            base_vexp_dV,
            base_vexp_V,
            K_vexp_dV,
            K_vexp_V)
    
    Base_GIFs.append(GIF_tmp)
    KCond_GIFs.append(KGIF_tmp)
    
dV_p.set_ylim(-5, 105)
dV_p.set_xlim(-0.5, 1.5)
V_p.set_ylim(-5, 105)
V_p.set_xlim(-0.5, 1.5)

plt.tight_layout()
plt.show()
    
print '\nDone!\n'


#%% PLOT FIT

print 'PLOTTING FIT ON TRAINING SET'
for i in range(len(experiments)):
    
    Base_GIFs[i].plotFit('Base GIF {}'.format(i))
    
    KCond_GIFs[i].plotFit('KCond GIF {}'.format(i))


#%% PLOT POWER SPECTRUM DENSITY

print 'EXTRACTING/PLOTTING POWER SPECTRUM DENSITY'
for i in range(len(experiments)):
    
    print '\rExtracting cell {}'.format(i),
    
    Base_GIFs[i].plotPowerSpectrumDensity('Base GIF {}'.format(i))
    
    KCond_GIFs[i].plotPowerSpectrumDensity('KCond GIF {}'.format(i))
    