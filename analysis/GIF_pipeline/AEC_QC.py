"""Active electrode compensation and quality control.

Ingest raw recordings in the form of ABF files, apply AEC, and sort cells into
`good` and `bad` according to intrinsic reliability and amount of firing rate
drift.

Requires an options JSON file with the following fields:

- AEC_col (str) index column to use for loading AEC files.
- train_cols (list) index column(s) to use for loading training data.
- test_cols (list) index column(s) to use for loading test data.
- no_test_traces (int) fail experiments without this number of test traces.
- train_ROIs (list of pairs) ROIs to use for training traces (see grr.Trace).
- test_ROIs (list of pairs) ROIs to use for test traces (see grr.Trace).
- drift_cutoff (float 0-1) fail experiments with high test set firing rate drift.
- reliability_cutoff (float 0-1) fail experiments with low intrinsic reliability.
- dt (float) timestep width in ms.

"""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import warnings
import sys

import numpy as np
import pandas as pd
from scipy import stats

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.SpikeTrainComparator import intrinsic_reliability
from grr.Tools import gagProcess


# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

# Paths to inputs and outputs.
parser.add_argument(
    'indexfile',
    help='Path to CSV spreadsheet with data file names.'
)
parser.add_argument(
    'goodcells',
    help='Path to save preprocessed QC-passing recordings.'
)
parser.add_argument(
    'opts',
    help='Path to options JSON file.'
)
parser.add_argument(
    '--badcells', default=None,
    help='Path to save preprocessed QC-failing recordings. '
    'Omit to skip.'
)
parser.add_argument(
    '--datafiles', default=None,
    help='Path to raw data files, if different from path to index file.'
)
parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Print information about progress.'
)

# Parse commandline args.
args = parser.parse_args()

# Get data from same folder as indexfile by default.
if args.datafiles is None:
    args.datafiles = os.path.dirname(args.indexfile)

# Load opts from JSON file.
with open(args.opts, 'r') as f:
    opts = json.load(f)
    f.close()

required_json_fields = ['AEC_col', 'train_cols', 'test_cols', 'no_test_traces',
                        'train_ROIs', 'test_ROIs', 'drift_cutoff',
                        'reliability_cutoff', 'dt']
for field in required_json_fields:
    if field not in opts:
        raise AttributeError(
            '{optsfile} missing required field {fieldname}'.format(
                optsfile=args.opts, fieldname=field
            )
        )

# READ IN DATA

file_index = pd.read_csv(args.indexfile)

experiments = []
for i in range(file_index.shape[0]):
    if args.verbose:
        print('Loading traces {:.1f}%'.format(100 * i / file_index.shape[0]), end='\r')
        sys.stdout.flush()

    try:
        with gagProcess():  # Silence verbose load methods.

            tmp_experiment = Experiment(file_index.loc[i, 'Cell'], opts['dt'])

            # Add AEC, training, and test traces.
            tmp_experiment.setAECTrace(
                'Axon',
                fname=os.path.join(
                    args.datafiles, file_index.loc[i, opts['AEC_col']]
                ),
                V_channel=0,
                I_channel=1
            )
            for traincol in opts['train_cols']:
                tmp_experiment.addTrainingSetTrace(
                    'Axon',
                    fname=os.path.join(
                        args.datafiles, file_index.loc[i, traincol]
                    ),
                    V_channel=0,
                    I_channel=1
                )
            for testcol in opts['test_cols']:
                tmp_experiment.addTestSetTrace(
                    'Axon',
                    fname=os.path.join(
                        args.datafiles, file_index.loc[i, testcol]
                    ),
                    V_channel=0,
                    I_channel=1
                )

            # Set trace ROIs.
            for tr in tmp_experiment.trainingset_traces:
                tr.setROI(opts['train_ROIs'])
            for tr in tmp_experiment.testset_traces:
                tr.setROI(opts['test_ROIs'])

        # Add loaded experiment to list.
        experiments.append(tmp_experiment)

    except RuntimeError:
        # Seems to be due to an issue with units expected by Experiment._readABF().
        # Probably a data problem rather than code problem.
        warnings.warn(
            'Problem with {} import. Skipping.'.format(
                file_index.loc[i, 'Cell']
            )
        )

if args.verbose:
    print('\nDone loading traces!\n')


# PERFORM AEC

for i, expt in enumerate(experiments):
    if args.verbose:
        print('Running AEC {:.1f}%'.format(100 * i / len(experiments)), end='\r')
        sys.stdout.flush()

    with gagProcess():

        tmp_AEC = AEC_Badel(expt.dt)

        tmp_AEC.K_opt.setMetaParameters(
            length=150.0, binsize_lb=expt.dt, binsize_ub=2.0,
            slope=30.0, clamp_period=1.0
        )
        tmp_AEC.p_expFitRange = [3.0, 150.0]
        tmp_AEC.p_nbRep = 15

        # Assign tmp_AEC to expt and compensate the voltage recordings
        expt.setAEC(tmp_AEC)
        expt.performAEC()

if args.verbose:
    print('\nDone running AEC!\n')


# DETECT SPIKES

for i, expt in enumerate(experiments):
    if args.verbose:
        print('Detecting spikes {:.1f}%'.format(100 * i / len(experiments)), end='\r')
        sys.stdout.flush()
    for tr in expt.trainingset_traces:
        tr.detectSpikes()
    for tr in expt.testset_traces:
        tr.detectSpikes()

if args.verbose:
    print('\nDone detecting spikes!\n')

# %% EXCLUDE BASED ON DRIFT IN NO. SPIKES

if args.verbose:
    print("Excluding cells with drift > {}.".format(opts['drift_cutoff']))

drifting_cells = []

for i, expt in enumerate(experiments):
    if len(expt.testset_traces) != opts['no_test_traces']:
        print('{:>16}Wrong no. of traces. Skipping...'.format(''))
        drifting_cells.append(i)  # Exclude it.
        continue

    spks = []
    for j in range(len(expt.testset_traces)):
        spks.append(expt.testset_traces[j].spks)

    no_spks_per_sweep = [len(s_) for s_ in spks]

    r, p = stats.pearsonr(
        no_spks_per_sweep,
        np.arange(0, opts['no_test_traces'])
    )

    if np.abs(r) > opts['drift_cutoff']:
        drifting_cells.append(i)

    if args.verbose:
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

        print('{:>2}    {}    R = {:>6.3f}, p = {:>5.3f}   {}'.format(i, expt.name, r, p, stars))


# %% EXCLUDE BASED ON INTRINSIC RELIABILITY

if args.verbose:
    print('Excluding cells with intrinsic reliability < {}'.format(opts['reliability_cutoff']))

unreliable_cells = []
reliability_ls = []
for i, expt in enumerate(experiments):

    try:
        reliability_tmp = intrinsic_reliability(expt.testset_traces, 8, 0.1)
        reliability_ls.append(reliability_tmp)

        if reliability_tmp < opts['reliability_cutoff']:
            unreliable_cells.append(i)
            stars = '*'
        else:
            stars = ''

        if args.verbose:
            print('{:>2}    {} IR = {:.3f} {}'.format(
                i, expt.name, reliability_tmp, stars)
            )
    except ValueError:
        warnings.warn('Problem with experiment {}, {}'.format(i, expt.name))

if args.verbose:
    print('Excluded {} cells due to low reliability.\n'.format(len(unreliable_cells)))


# %% REMOVE EXCLUDED CELLS FROM DATASET AND PICKLE RESULT

# Construct list of cells to exclude.
bad_cell_inds = []
[bad_cell_inds.extend(x) for x in [drifting_cells, unreliable_cells]]
bad_cell_inds = np.unique(bad_cell_inds)

# Pop bad cells out of experiments and place in bad_cells list.
bad_cells = []
for i in np.flip(np.sort(bad_cell_inds), -1):
    bad_cells.append(experiments.pop(i))

print('Excluding {}/{} cells.'.format(len(bad_cells), len(bad_cells) + len(experiments)))

if args.verbose:
    print('Saving {} good cells to {}'.format(len(experiments), args.goodcells))
with open(args.goodcells, 'wb') as f:
    pickle.dump(experiments, f)
    f.close()

if args.badcells is not None:
    if args.verbose:
        print('Saving {} bad cells to {}'.format(len(bad_cells), args.badcells))
    with open(args.badcells, 'wb') as f:
        pickle.dump(bad_cells, f)
        f.close()

print('Finished! Exiting.')
