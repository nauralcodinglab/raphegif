#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse

import numpy as np

from src.SpikeTrainComparator import intrinsic_reliability


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'Experiments', help = 'Path to pickled dict of Experiment objects.'
)
parser.add_argument(
    'output', help = 'Path to save output.'
)
parser.add_argument(
    '--minsize',
    help = 'Smallest intrinsic reliability window to try (ms).',
    type = float, default = 1.
)
parser.add_argument(
    '--maxsize',
    help = 'Largest intrinsic reliability window to try (ms).',
    type = float, default = 100.
)
parser.add_argument(
    '--num',
    help = 'Number of intrinsic reliability windows to try.',
    type = int, default = 30
)
parser.add_argument(
    '-v', '--verbose',
    help = 'Print information about progress.',
    action = 'store_true'
)
args = parser.parse_args()


#%% LOAD EXPERIMENTS

if args.verbose:
    print 'Loading experiments from {}'.format(args.Experiments)
with open(args.Experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()


#%% COMPUTE RELIABILITIES

# Allocate dicts to store output.
reliabilities = {}
windows_arrs = {}

# Compute window sizes.
windows = np.logspace(
    np.log10(args.minsize),
    np.log10(args.maxsize),
    args.num
)

# Iterate over cell types.
for key in experiments.keys():
    reliabilities[key] = []

    # Iterate over window sizes.
    for i, win in enumerate(windows):
        if args.verbose:
            print 'Extracting {} reliability {:.1f}%'.format(
                key, 100*(i + 1)/len(windows)
            )

        # Iterate over experiments.
        reliabilities_tmp = []
        for expt in experiments[key]:
            reliabilities_tmp.append(intrinsic_reliability(expt.testset_traces, win, 0.1))

        reliabilities[key].append(reliabilities_tmp)

    reliabilities[key] = np.array(reliabilities[key])
    windows_arrs[key] = np.tile(windows[:, np.newaxis], (1, reliabilities[key].shape[1]))


#%% SAVE OUTPUT

if args.verbose:
    print 'Saving output to {}'.format(args.output)
output_dict = {'reliabilities': reliabilities, 'supports': windows_arrs}
with open(args.output, 'wb') as f:
    pickle.dump(output_dict, f)
    f.close()
if args.verbose:
    print 'Done!'

