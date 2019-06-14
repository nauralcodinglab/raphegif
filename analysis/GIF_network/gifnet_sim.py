"""Script for running GIFnet simulations.

INTENDED TO BE RUN FROM COMMAND LINE
"""

#%% IMPORT MODULES

from __future__ import division

import argparse
import pickle

import numpy as np

from src.Simulation import GIFnet_Simulation
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
    'output',
    help = 'Filepath for storing the results of the simulation.'
)
parser.add_argument(
    '--num-ser-examples',
    help = 'Number of 5HT example traces to save.',
    type = int, default = 20
)
parser.add_argument(
    '--num-gaba-examples',
    help = 'Number of GABA example traces to save.',
    type = int, default = 20
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

# Background noise params.
# For efficiency, same noise is used for all sweeps.
parser.add_argument(
    '--no-noise', help = 'Turn off background network noise.',
    action = 'store_true'
)
parser.add_argument(
    '--tau-background',
    help = 'Time constant of background noise (ms).',
    type = float, default = 3.
)
parser.add_argument(
    '--sigma-background',
    help = 'Spread of background noise (nA).',
    type = float, default = 0.005
)
parser.add_argument(
    '--seed-background',
    help = 'Seed for background noise random number generator.',
    type = int, default = 42
)

# Misc.
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

if 'ser_input' in distal_in.keys():
    if distal_in['ser_input'].ndim == 1:
        distal_in['ser_input'] = np.broadcast_to(
            distal_in['ser_input'],
            (1, gifnet_mod.no_ser_neurons, distal_in['ser_input'].shape[0])
        )
    elif distal_in['ser_input'].ndim == 3:
        distal_in['ser_input'] = np.broadcast_to(
            distal_in['ser_input'],
            (distal_in['ser_input'].shape[0],
            gifnet_mod.no_ser_neurons,
            distal_in['ser_input'].shape[2])
        )
    else:
        raise ValueError(
            'ser_input must be a 1D or 3D array'
        )

if 'gaba_input' in distal_in.keys():
    if distal_in['gaba_input'].ndim == 1:
        distal_in['gaba_input'] = np.broadcast_to(
            distal_in['gaba_input'],
            (1, gifnet_mod.no_gaba_neurons, distal_in['gaba_input'].shape[0])
        )
    elif distal_in['gaba_input'].ndim == 3:
        distal_in['gaba_input'] = np.broadcast_to(
            distal_in['gaba_input'],
            (distal_in['gaba_input'].shape[0],
            gifnet_mod.no_gaba_neurons,
            distal_in['gaba_input'].shape[2])
        )
    else:
        raise ValueError(
            'gaba_input must be a 1D or 3D array'
        )


#%% OPTIONALLY, ADD BACKGROUND NOISE TO INPUT

if not args.no_noise:

    if args.verbose:
        print 'Adding background noise.'

    for i, input_type in enumerate(distal_in.keys()):

        # Ensure input_type is allowable.
        # (distal_in might also have a key for metaparams)
        if input_type not in ['ser_input', 'gaba_input']:
            continue

        # Copy distal_in to ensure it is writable.
        distal_in[input_type] = np.copy(distal_in[input_type])

        # Generate random input for each cell.
        np.random.seed(args.seed_background + i)
        for cell_no in range(distal_in[input_type].shape[1]):

            # Generate background noise.
            tmp_bg_noise = generateOUprocess(
                int(distal_in[input_type].shape[2] * gifnet_mod.dt),
                args.tau_background,
                0., args.sigma_background,
                gifnet_mod.dt,
                None
            ).astype(np.float32)

            # Broadcast bg noise to right shape and add.
            distal_in[input_type][:, cell_no, :] += np.broadcast_to(
                tmp_bg_noise,
                (distal_in[input_type].shape[0],
                distal_in[input_type].shape[2])
            )


#%% RUN SIMULATION

meta_args = {
    'name': getattr(gifnet_mod, 'name', 'Untitled'),
    'dt': gifnet_mod.dt
}

# Simulations without 5HT neurons.
if args.no_ser:
    if args.verbose:
        print(
            'Running simulations without 5HT neurons '
            'and saving output to {}'.format(args.output)
        )

    # Update metaparameters.
    meta_args.update({
        'T': int(distal_in['gaba_input'].shape[2] * gifnet_mod.dt),
        'no_sweeps': distal_in['gaba_input'].shape[0],
        'no_ser_examples': 0,
        'no_gaba_examples': args.num_gaba_examples
    })

    # Run and save simulation simultaneously.
    with GIFnet_Simulation(args.output, **meta_args) as outfile:
        # Set channels to save in examples.
        outfile.init_gaba_examples()

        # Run simulation.
        gifnet_mod.simulate(
            outfile,
            gaba_input = distal_in['gaba_input'],
            do_feedforward = ~args.no_feedforward, verbose = args.verbose
        )

        outfile.close()

    if args.verbose:
        print 'Done!'

# Simulations without GABA neurons.
elif args.no_gaba:
    if args.verbose:
        print(
            'Running simulations without GABA neurons '
            'and saving output to {}'.format(args.output)
        )

    # Update metaparameters.
    meta_args.update({
        'T': int(distal_in['ser_input'].shape[2] * gifnet_mod.dt),
        'no_sweeps': distal_in['ser_input'].shape[0],
        'no_ser_examples': args.num_ser_examples,
        'no_gaba_examples': 0
    })

    # Run and save simulation simultaneously.
    with GIFnet_Simulation(args.output, **meta_args) as outfile:
        # Set channels to save in examples.
        outfile.init_ser_examples()

        # Run simulation.
        gifnet_mod.simulate(
            outfile,
            ser_input = distal_in['ser_input'],
            do_feedforward = ~args.no_feedforward, verbose = args.verbose
        )

        outfile.close()

    if args.verbose:
        print 'Done!'

# Run full simulations.
else:
    if args.verbose:
        print(
            'Running simulations and saving output '
            'to {}'.format(args.output)
        )

    # Update metaparameters.
    meta_args.update({
        'T': int(distal_in['ser_input'].shape[2] * gifnet_mod.dt),
        'no_sweeps': distal_in['ser_input'].shape[0],
        'no_ser_examples': args.num_ser_examples,
        'no_gaba_examples': args.num_gaba_examples
    })

    # Run and save simulation simultaneously.
    with GIFnet_Simulation(args.output, **meta_args) as outfile:
        # Set channels to save in examples.
        outfile.init_ser_examples()
        outfile.init_gaba_examples()

        # Run simulation.
        gifnet_mod.simulate(
            outfile,
            ser_input = distal_in['ser_input'],
            gaba_input = distal_in['gaba_input'],
            do_feedforward = ~args.no_feedforward,
            verbose = args.verbose
        )

        outfile.close()

    if args.verbose:
        print 'Done!'

