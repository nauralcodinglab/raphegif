#%% IMPORT MODULES

from __future__ import division

import os
import pickle
import argparse
import warnings

from grr.Tools import gagProcess


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'experiments',
    help = 'Pickled list of Experiment objects'
)
parser.add_argument(
    'models',
    help = 'Pickled list of GIF or GIF-like models to use to compute spiketrain predictions.'
)
parser.add_argument(
    'output',
    help = 'Path to use for pickling output.'
)
parser.add_argument(
    '-v', '--verbose', help = 'Print more stuff.',
    action = 'store_true'
)
parser.add_argument(
    '--precision', help = 'Width of window to consider coincidence (ms).',
    default = 4., type = float
)
parser.add_argument(
    '--repeats', help = 'Number of spiketrains to realize from model.',
    default = 500, type = int
)
args = parser.parse_args()


#%% LOAD MODELS/DATA

if args.verbose:
    print 'Loading experiments from {}'.format(args.experiments)
with open(args.experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()

if args.verbose:
    print 'Loading models from {}'.format(args.models)
with open(args.models, 'rb') as f:
    models = pickle.load(f)
    f.close()

# Check that number of experiments and models match.
if len(experiments) != len(models):
    warnings.warn(
        'Unequal number of experiments ({}) and models ({}).'.format(len(experiments), len(models)),
        RuntimeWarning
    )


#%% COMPUTE MD*

Md_vals = []
predictions = []
names = []

# Loop over cells.
if args.verbose:
    print 'Computing Md*...'
for expt in experiments:

    # Select matching model.
    matched_mod_flag = False
    for mod in models:
        if mod.name == expt.name:
            matched_mod_flag = True
            break
        else:
            continue

    # Handle case that no matching model is found.
    if not matched_mod_flag:
        warnings.warn(
            'No match found for experiment {}. Skipping.'.format(expt.name),
            RuntimeWarning
        )
        continue

    # Double check that correct mod was selected.
    assert mod.name == expt.name, 'Model and experiment do not match.'

    with gagProcess():

        # Use mod to predict the spiking data of the test data set from expt.
        tmp_prediction = expt.predictSpikes(mod, nb_rep=500)

        tmp_Md = tmp_prediction.computeMD_Kistler(args.precision, 0.1)

    predictions.append(tmp_prediction)
    Md_vals.append(tmp_Md)
    names.append(expt.name)

    if args.verbose:
        print '{} MD* {}ms: {:.2f}'.format(expt.name, args.precision, tmp_Md)


#%% SAVE OUTPUT

output_dict = {
    'Md_vals': Md_vals,
    'predictions': predictions,
    'names': names
}

if args.verbose:
    print 'Saving output to {}'.format(args.output)
with open(args.output, 'wb') as f:
    pickle.dump(output_dict, f)
    f.close()
if args.verbose:
    print 'Done!'

