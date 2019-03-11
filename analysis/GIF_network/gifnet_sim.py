"""Script for running parallelized GIFnet simulations.

INTENDED TO BE RUN FROM COMMAND LINE
"""

#%% IMPORT MODULES

from __future__ import division

import argparse
import pickle
import multiprocessing as mp

import numpy as np

import src.GIF_network as gfn

#%% PARSE COMMANDLINE ARGUMENTS

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
    '--input_tau', type = float,
    help = 'Time constant of synaptic input alpha kernel.',
    default = 15.
)
parser.add_argument(
    '--processes', type = int,
    help = 'Number of processes to spawn to accelerate simulations.',
    default = 4.
)

args = parser.parse_args()


#%% LOAD MODEL

with open(args.model, 'rb') as f:
    gifnet_mod = pickle.load(f)
    f.close()

distal_in = SynapticKernel(
    'alpha', tau = 15, ampli = 1, kernel_len = 500, dt = dt
).centered_kernel
