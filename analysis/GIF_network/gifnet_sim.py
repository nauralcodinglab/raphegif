"""Script for running GIFnet simulations.

INTENDED TO BE RUN FROM COMMAND LINE
"""

#%% IMPORT MODULES

from __future__ import division

import argparse
import pickle

import numpy as np

import src.GIF_network as gfn
from src.Tools import generateOUprocess


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
    help = 'Path to GIFnet model used to run the simulation.'
)
parser.add_argument(
    'input',
    help = 'Filepath to pickled dict with model input. '
    'Input must be in a dict with `ser_input` and `gaba_input` as '
    'keys. If inputs are vectors, they are broadcasted to match '
    'the number of neurons of each type.'
)
parser.add_argument(
    'destination_path',
    help = 'Filepath for storing the results of the simulation.'
)
parser.add_argument(
    '--no-ser',
    help = 'Do not simulate 5HT neurons, even if input has been provided.',
    action = 'store_true'
)
parser.add_argument(
    '--no-gaba',
    help = 'Do not simulate GABA neurons, even if input has been provided.',
    action = 'store_true'
)
parser.add_argument(
    '--no-feedforward',
    help = 'Passes do_feedforward = False to GIFnet.simulate. '
    'Allows 5HT and GABA cells to be simulated, but does not '
    'connect them.',
    action = 'store_true'
)
parser.add_argument(
    '-v', '--verbose',
    help = 'Increase output verbosity.',
    action = 'store_true'
)

args = parser.parse_args()


#%% LOAD MODEL AND INITIALIZE VARS NEEDED TO RUN SIMULATION

# Load model.
if args.verbose:
    print 'Loading GIFnet model from {}'.format(args.model)
with open(args.model, 'rb') as f:
    gifnet_mod = pickle.load(f)
    f.close()

# Load input.
if args.verbose:
    print 'Loading input from {}'.format(args.input)
with open(args.input, 'rb') as f:
    distal_in = pickle.load(f)
    f.close()
if hasattr(distal_in, 'ser_input') and distal_in['ser_input'].ndim == 1:
    distal_in['ser_input'] = np.broadcast_to(
        distal_in['ser_input'],
        (distal_in['ser_input'].shape[0], gifnet_mod.no_ser_neurons)
    )
if hasattr(distal_in, 'gaba_input') and distal_in['gaba_input'].ndim == 1:
    distal_in['gaba_input'] = np.broadcast_to(
        distal_in['gaba_input'],
        (distal_in['gaba_input'].shape[0], gifnet_mod.no_gaba_neurons)
    )


#%% RUN SIMULATION

if args.no_ser:
    if args.verbose:
        print 'Running simulations without 5HT neurons.'
    results = gifnet_mod.simulate(
        gaba_input = distal_in['gaba_input'], 
        do_feedforward = ~args.no_feedforward, verbose = args.verbose
    )
elif args.no_gaba:
    if args.verbose:
        print 'Running simulations without GABA neurons.'
    results = gifnet_mod.simulate(
        ser_input = distal_in['ser_input'], 
        do_feedforward = ~args.no_feedforward, verbose = args.verbose
    )
else:
    if args.verbose:
        print 'Running simulations.'
    results = gifnet_mod.simulate(
        **distal_in, 
        do_feedforward = ~args.no_feedforward, verbose = args.verbose
    )


#%% ADD METADATA TO RESULTS

results.set_metadata({'input': args.input, 'model': args.model})


#%% SAVE OUTPUT

if args.verbose:
    print 'Saving output to {}'.format(args.destination_path)
results.save_hdf(args.destination_path)
if args.verbose:
    print 'Done!'
