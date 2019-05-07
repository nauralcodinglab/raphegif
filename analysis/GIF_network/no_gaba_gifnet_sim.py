"""Script for running parallelized GIFnet simulations.

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        help = 'Path to GIFnet model used to run the simulation.'
    )
    parser.add_argument(
        'destination_path',
        help = 'Filepath for storing the results of the simulation.'
    )
    parser.add_argument(
        '-v', '--verbose',
        help = 'Increase output verbosity.',
        action = 'store_true'
    )

    args = parser.parse_args()


#%% LOAD MODEL AND INITIALIZE VARS NEEDED TO RUN SIMULATION

if __name__ == '__main__':

    ### Load model.
    with open(args.model, 'rb') as f:
        gifnet_mod = pickle.load(f)
        f.close()

    distal_in = { 
    'ser_input': np.array([generateOUprocess(60000., 100., 0.050, 0.050, 0.1, 42).astype(np.float32)] * 1200, dtype = np.float32),
    'gaba_input': np.array([generateOUprocess(60000, 30., 0.050, 0.050, 0.1, 43).astype(np.float32)] * 1, dtype = np.float32)
    }


#%% RUN SIMULATION

if args.verbose:
    print 'Running simulations...'
results = gifnet_mod.simulate(distal_in)


#%% SAVE OUTPUT

if args.verbose:
    print 'Saving output to {}'.format(args.destination_path)
with open(args.destination_path, 'wb') as f:
    pickle.dump(results, f)
    f.close()
if args.verbose:
    print 'Done!'
