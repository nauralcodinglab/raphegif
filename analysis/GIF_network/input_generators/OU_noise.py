#%% IMPORT MODULES

from __future__ import division

import os
import pickle
import argparse
import copy

import numpy as np

from grr.Tools import generateOUprocess

#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument(
    'output', help = 'Path to save output. '
    'Pickled dict with ser_input and gaba_input as attributes.'
)

# 5HT parameters.
parser.add_argument(
    '--tau-ser', type = float, default = 30.,
    help = 'Time constant of 5HT input noise (ms).'
)
parser.add_argument(
    '--mean-ser', type = float, default = 0.050,
    help = 'Mean of 5HT input noise (nA).'
)
parser.add_argument(
    '--sigma-ser', type = float, default = 0.050,
    help = 'Spread of 5HT input noise (nA).'
)
parser.add_argument(
    '--seed-ser', type = float, default = 42,
    help = 'Random seed for 5HT input noise.'
)

# GABA parameters.
parser.add_argument(
    '--copy-ser-params', action = 'store_true',
    help = 'Use 5HT noise params for GABA noise. '
    'Overries any GABA noise params that have been set.'
)
parser.add_argument(
    '--tau-gaba', type = float, default = 30.,
    help = 'Time constant of GABA input noise (ms).'
)
parser.add_argument(
    '--mean-gaba', type = float, default = 0.050,
    help = 'Mean of GABA input noise (nA).'
)
parser.add_argument(
    '--sigma-gaba', type = float, default = 0.050,
    help = 'Spread of GABA input noise (nA).'
)
parser.add_argument(
    '--seed-gaba', type = float, default = 42,
    help = 'Random seed for GABA input noise.'
)
parser.add_argument(
    '-v', '--verbose',
    help = 'Print more information.',
    action = 'store_true'
)

# General parameters.
parser.add_argument(
    '-d', '--duration',
    type = float, default = 60000.,
    help = 'Duration of noise to realize.'
)
parser.add_argument(
    '--dt', type = float, default = 0.1,
    help = 'Timestep of noise.'
)

args = parser.parse_args()


#%% GENERATE NOISE TO EXPORT

if args.verbose:
    print 'Generating noise.'
distal_input = {}
distal_input['metaparams'] = {
    'input_type': 'OU_noise',
    'duration': args.duration,
    'dt': args.dt
}

# Generate ser_input.
distal_input['ser_input'] = generateOUprocess(
    args.duration, args.tau_ser,
    args.mean_ser, args.sigma_ser,
    args.dt, args.seed_ser
).astype(np.float32)

distal_input['metaparams'].update({
    'ser_tau': args.tau_ser,
    'ser_mean': args.mean_ser,
    'ser_sigma': args.sigma_ser,
    'ser_seed': args.seed_ser
})

# Generate gaba_input.
if args.copy_ser_params:
    distal_input['gaba_input'] = copy.deepcopy(distal_input['ser_input'])

    distal_input['metaparams'].update({
        'gaba_tau': args.tau_ser,
        'gaba_mean': args.mean_ser,
        'gaba_sigma': args.sigma_ser,
        'gaba_seed': args.seed_ser
    })

else:
    distal_input['gaba_input'] = generateOUprocess(
        args.duration, args.tau_gaba,
        args.mean_gaba, args.sigma_gaba,
        args.dt, args.seed_gaba
    ).astype(np.float32)

    distal_input['metaparams'].update({
        'gaba_tau': args.tau_gaba,
        'gaba_mean': args.mean_gaba,
        'gaba_sigma': args.sigma_gaba,
        'gaba_seed': args.seed_gaba
    })


#%% SAVE NOISE

if args.verbose:
    print 'Saving noise to {}'.format(args.output)
with open(args.output, 'wb') as f:
    pickle.dump(distal_input, f)
    f.close()
if args.verbose:
    print 'Done!'

