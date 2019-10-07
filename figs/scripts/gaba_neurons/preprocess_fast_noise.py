#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.SpikeTrainComparator import intrinsic_reliability
from grr.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = os.path.join('data', 'GABA_cells')

file_index = pd.read_csv(os.path.join(DATA_PATH, 'index.csv'))

experiments = []

for i in range(file_index.shape[0]):

    try:
        with gagProcess():

            tmp_experiment = Experiment(file_index.loc[i, 'Cell'], 0.1)
            tmp_experiment.setAECTrace(
                'Axon', fname = os.path.join(DATA_PATH, file_index.loc[i, 'AEC2']),
                V_channel = 0, I_channel = 1
            )

            for ind in ['1', '2', '3']:

                tmp_experiment.addTrainingSetTrace(
                    'Axon', fname = os.path.join(DATA_PATH, file_index.loc[i, 'Train' + ind]),
                    V_channel = 0, I_channel = 1
                )
                tmp_experiment.addTestSetTrace(
                    'Axon', fname = os.path.join(DATA_PATH, file_index.loc[i, 'Test' + ind]),
                    V_channel = 0, I_channel = 1
                )

        experiments.append(tmp_experiment)

    except RuntimeError:
        # Seems to be due to an issue with units expected by Experiment._readABF().
        # Probably a data problem rather than code problem.
        print 'Problem with {} import. Skipping.'.format(file_index.loc[i, 'Cell'])


#%% PERFORM AEC

AECs = []

for expt in experiments:

    with gagProcess():

        tmp_AEC = AEC_Badel(expt.dt)

        tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
        tmp_AEC.p_expFitRange = [3.0,150.0]
        tmp_AEC.p_nbRep = 15

        # Assign tmp_AEC to expt and compensate the voltage recordings
        expt.setAEC(tmp_AEC)
        expt.performAEC()

    AECs.append(tmp_AEC)


#%% DETECT SPIKES

for expt in experiments:

    for tr in expt.trainingset_traces:
        tr.detectSpikes()

    for tr in expt.testset_traces:
        tr.detectSpikes()


#%% EXCLUDE BASED ON DRIFT IN NO. SPIKES

print "Excluding cells based on drift."

drifting_cells = []

for i, expt in enumerate(experiments):
    if len(expt.testset_traces) != 9:
        print('{:>16}Wrong no. of traces. Skipping...'.format(''))
        drifting_cells.append(i) # Exclude it anyway.
        continue

    spks = []
    for j in range(len(expt.testset_traces)):
        spks.append(expt.testset_traces[j].spks)

    no_spks_per_sweep = [len(s_) for s_ in spks]

    r, p = stats.pearsonr(no_spks_per_sweep, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    if p > 0.1:
        stars = ''
    elif p > 0.05:
        stars = 'o'
    elif p > 0.01:
        stars = '*'
    elif p > 0.001:
        stars = '**'
    else:
        stars = '***'

    if np.abs(r) > 0.8:
        drifting_cells.append(i)

    print('{:>2}    {}    R = {:>6.3f}, p = {:>5.3f}   {}'.format(i, expt.name, r, p, stars))


#%% EXCLUDE BASED ON INTRINSIC RELIABILITY

print 'Excluding cells based on intrinsic reliability...'

unreliable_cells = []
reliability_ls = []
for i, expt in enumerate(experiments):

    try:
        reliability_tmp = intrinsic_reliability(expt.testset_traces, 8, 0.1)
        reliability_ls.append(reliability_tmp)

        if reliability_tmp < 0.2:
            unreliable_cells.append(i)
            stars = '*'
        else:
            stars = ''

        print('{:>2}    {} IR = {:.3f} {}'.format(
            i, expt.name, reliability_tmp, stars)
        )
    except ValueError:
        print 'Problem with experiment {}'.format(i)

print 'Excluded {} cells due to low reliability.'.format(len(unreliable_cells))


#%% REMOVE EXCLUDED CELLS FROM DATASET AND PICKLE RESULT

bad_cell_inds = []
[bad_cell_inds.extend(x) for x in [drifting_cells, unreliable_cells]] 
bad_cell_inds = np.unique(bad_cell_inds)

bad_cells = []

for i in np.flip(np.sort(bad_cell_inds), -1):
    bad_cells.append(experiments.pop(i))

print 'Excluding {}/{} cells.'.format(len(bad_cells), len(bad_cells) + len(experiments))

PROCESSED_PATH = os.path.join('data', 'processed', 'GABA_fastnoise', 'gaba_goodcells.ldat')
print 'Saving {} `good` cells to {}'.format(len(experiments), PROCESSED_PATH)

with open(PROCESSED_PATH, 'wb') as f:
    pickle.dump(experiments, f)
    f.close()

print 'Done!'

