import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np
from ezephys import stimtools

# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument(
    'output',
    help='Path to save output. '
    'Pickled dict with ser_input and gaba_input as attributes.',
)

parser.add_argument('--dt', type=float, default=0.1, help='Timestep of input.')
parser.add_argument('--baseline', type=float, default=0.0, help="Baseline (constant) input.")
parser.add_argument(
    '-v',
    '--verbose',
    help='Print more information about progress.',
    action='store_true',
)

args = parser.parse_args()


# ASSEMBLE STIMULUS

stimulus_parameters = {
    'amplitude': 0.1,
    'pad_duration': 2000.  # Time to leave between stimuli.
}

pad_stimulus = stimtools.StepStimulus(
    [stimulus_parameters['pad_duration']], [0.0], args.dt
)

cos_wave = stimtools.CosStimulus(
    stimulus_parameters['amplitude'] / 2.,
    -stimulus_parameters['amplitude'] / 2.,
    1.0 / 4.0,
    4000.0,
    args.dt,
)

short_step = stimtools.StepStimulus(
    [200.0], [stimulus_parameters['amplitude']], dt=args.dt
)

step = stimtools.StepStimulus(
    [2000.0], [stimulus_parameters['amplitude']], args.dt
)

full_stimulus = stimtools.concatenate([
    pad_stimulus,
    cos_wave,
    pad_stimulus,
    short_step,
    pad_stimulus,
    step,
    pad_stimulus
])

wave_stimulus = {
    'ser_input': full_stimulus.command[np.newaxis, np.newaxis, :] + args.baseline,
    'gaba_input': full_stimulus.command[np.newaxis, np.newaxis, :] + args.baseline,
    'metaparams': {}
}

if args.verbose:
    print('Saving wave stimulus to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(wave_stimulus, f)
    f.close()
if args.verbose:
    print('Done!')
