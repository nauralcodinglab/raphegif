#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse

import numpy as np

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
from FeedForwardDRN import SynapticKernel
from grr.Tools import generateOUprocess

#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument(
    'output', help = 'Path to save output. '
    'Pickled dict with ser_input and gaba_input as attributes.'
)

# 5HT parameters.
parser.add_argument(
    '--min-ser', type = float, default = 0.010,
    help = 'Amplitude of smallest synaptic pulse to 5HT cells (nA).'
)
parser.add_argument(
    '--max-ser', type = float, default = 0.100,
    help = 'Amplitude of largest synaptic pulse to 5HT cells (nA).'
)
parser.add_argument(
    '--tau-ser', type = float, default = 10.,
    help = 'Time constant of 5HT synaptic input pulse (ms).'
)
parser.add_argument(
    '--baseline-ser', type = float, default = 0.,
    help = '5HT steady state input (nA).'
)
parser.add_argument(
    '--time-ser', type = float, default = 1000.,
    help = 'Time of onset of synaptic pulse to 5HT neurons.'
)
parser.add_argument(
    '--tau-ser-background', type = float, default = 3.,
    help = 'Time constant of background OU noise input to 5HT cells.'
)
parser.add_argument(
    '--sigma-ser-background', type = float, default = 0.,
    help = 'Spread of background OU noise input to 5HT cells. '
    'Set to zero to skip.'
)
parser.add_argument(
    '--seed-ser-background', type = int, default = 42,
    help = 'Seed for random number generator for background OU '
    'noise input to 5HT cells.'
)

# GABA parameters.
parser.add_argument(
    '--min-gaba', type = float, default = 0.010,
    help = 'Amplitude of smallest synaptic pulse to GABA cells (nA).'
)
parser.add_argument(
    '--max-gaba', type = float, default = 0.100,
    help = 'Amplitude of largest synaptic pulse to GABA cells (nA).'
)
parser.add_argument(
    '--tau-gaba', type = float, default = 10.,
    help = 'Time constant of GABA synpatic input pulse (ms).'
)
parser.add_argument(
    '--baseline-gaba', type = float, default = 0.,
    help = 'GABA steady state input (nA).'
)
parser.add_argument(
    '--time-gaba', type = float, default = 1000.,
    help = 'Time of onset of synaptic pulse to GABA neurons.'
)
parser.add_argument(
    '--tau-gaba-background', type = float, default = 3.,
    help = 'Time constant of background OU noise input to GABA cells.'
)
parser.add_argument(
    '--sigma-gaba-background', type = float, default = 0.,
    help = 'Spread of background OU noise input to GABA cells. '
    'Set to zero (default) to skip.'
)
parser.add_argument(
    '--seed-gaba-background', type = int, default = 43,
    help = 'Seed for random number generator for background OU '
    'noise input to GABA cells.'
)

# General parameters.
parser.add_argument(
    '-s', '--sweeps', type = int, default = 10,
    help = 'Number of sweeps to get from min to max amplitude.'
)
parser.add_argument(
    '-d', '--duration',
    type = float, default = 2000.,
    help = 'Duration of input to realize.'
)
parser.add_argument(
    '--dt', type = float, default = 0.1,
    help = 'Timestep of input.'
)
parser.add_argument(
    '-v', '--verbose',
    help = 'Print more information about progress.',
    action = 'store_true'
)

args = parser.parse_args()


#%% GENERATE SYNAPTIC PULSES TO EXPORT

if args.verbose:
    print 'Generating synaptic pulses.'
distal_input = {}
distal_input['metaparams'] = {
    'input_type': 'alpha_synaptic_pulse',
    'sweeps': args.sweeps,
    'duration': args.duration,
    'dt': args.dt
}

### Generate ser_input.
# Generate kernel and support.
ser_kernel = SynapticKernel(
    'alpha', tau = args.tau_ser, ampli = 1.,
    kernel_len = args.tau_ser * 10., dt = args.dt
).centered_kernel.astype(np.float32)

ser_support = np.zeros(
    int(args.duration / args.dt + 0.5),
    dtype = np.float32
)
ser_support[int(args.time_ser / args.dt)] = 1

# Convolve, tile, and check shape.
ser_convolved = np.tile(
    np.convolve(ser_support, ser_kernel, 'same'),
    (args.sweeps, 1, 1)
)
assert ser_convolved.shape == (args.sweeps, 1, len(ser_support))

# Scale pulses, then offset.
ser_convolved *= np.linspace(
    args.min_ser, args.max_ser, args.sweeps
)[:, np.newaxis, np.newaxis]
ser_convolved += args.baseline_ser

# Add noise.
if args.sigma_ser_background != 0.:
    if args.verbose:
        print 'Adding noise to ser pulses.'

    np.random.seed(args.seed_ser_background)
    for sweep_no in range(ser_convolved.shape[0]):
        ser_convolved[sweep_no, 0, :] += generateOUprocess(
            args.duration, args.tau_ser_background,
            0., args.sigma_ser_background,
            args.dt,
            None
        ).astype(np.float32)

# Add to output dict.
distal_input['ser_input'] = ser_convolved

distal_input['metaparams'].update({
    'ser_tau': args.tau_ser,
    'ser_baseline': args.baseline_ser,
    'ser_time': args.time_ser,
    'ser_min': args.min_ser,
    'ser_max': args.max_ser,
    'ser_background_tau': args.tau_ser_background,
    'ser_background_sigma': args.sigma_ser_background,
    'ser_background_seed': args.seed_ser_background
})

### Generate gaba_input.
# Generate kernel and support.
gaba_kernel = SynapticKernel(
    'alpha', tau = args.tau_gaba, ampli = 1.,
    kernel_len = args.tau_gaba * 10., dt = args.dt
).centered_kernel.astype(np.float32)

gaba_support = np.zeros(
    int(args.duration / args.dt + 0.5),
    dtype = np.float32
)
gaba_support[int(args.time_gaba / args.dt)] = 1

# Convolve, tile, and check shape.
gaba_convolved = np.tile(
    np.convolve(gaba_support, gaba_kernel, 'same'),
    (args.sweeps, 1, 1)
)
assert gaba_convolved.shape == (args.sweeps, 1, len(gaba_support))

# Scale pulses, then offset.
gaba_convolved *= np.linspace(
    args.min_gaba, args.max_gaba, args.sweeps
)[:, np.newaxis, np.newaxis]
gaba_convolved += args.baseline_gaba

# Add noise.
if args.sigma_gaba_background != 0.:
    if args.verbose:
        print 'Adding noise to gaba pulses.'

    np.random.seed(args.seed_gaba_background)
    for sweep_no in range(gaba_convolved.shape[0]):
        gaba_convolved[sweep_no, 0, :] += generateOUprocess(
            args.duration, args.tau_gaba_background,
            0., args.sigma_gaba_background,
            args.dt,
            None
        ).astype(np.float32)

# Add to output dict.
distal_input['gaba_input'] = gaba_convolved

distal_input['metaparams'].update({
    'gaba_tau': args.tau_gaba,
    'gaba_baseline': args.baseline_gaba,
    'gaba_time': args.time_gaba,
    'gaba_min': args.min_gaba,
    'gaba_max': args.max_gaba,
    'gaba_background_tau': args.tau_gaba_background,
    'gaba_background_sigma': args.sigma_gaba_background,
    'gaba_background_seed': args.seed_gaba_background
})

#%% SAVE SYNAPTIC PULSES

if args.verbose:
    print 'Saving synaptic pulses to {}'.format(args.output)
with open(args.output, 'wb') as f:
    pickle.dump(distal_input, f)
    f.close()
if args.verbose:
    print 'Done!'

