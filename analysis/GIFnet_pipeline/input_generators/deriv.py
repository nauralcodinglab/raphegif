import pickle
import argparse

import numpy as np
from ezephys import stimtools as st

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

MAX_RAMP_DURATION = 1e4
NUM_SWEEPS = 20
stimulus_parameters = {
    'amplitude': 0.1,
    'ramp_durations': np.logspace(np.log10(100), np.log10(MAX_RAMP_DURATION), NUM_SWEEPS),
}

sweeps = []
for duration in stimulus_parameters['ramp_durations']:
    sweeps.append(
        st.concatenate(
            [
                st.StepStimulus([2e3], [0], args.dt), 
                np.linspace(0, stimulus_parameters['amplitude'], duration / args.dt), 
                st.StepStimulus([MAX_RAMP_DURATION - duration + 500], [0], args.dt)
            ], 
            args.dt
        ).command[np.newaxis, :124990]
    )

sweeps = np.array(sweeps)
assert np.shape(sweeps)[:2] == (NUM_SWEEPS, 1)

ramp_stimulus = {
    'ser_input': sweeps + args.baseline,
    'gaba_input': sweeps + args.baseline,
    'metaparams': {
        'deriv_in_nA_s': (
            stimulus_parameters['amplitude'] 
            / (stimulus_parameters['ramp_durations'] * 1e-3)
        ),
        'dt': args.dt,
        'baseline': args.baseline,
        'amplitude': stimulus_parameters['amplitude'],
    }
}

if args.verbose:
    print('Saving wave stimulus to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(ramp_stimulus, f)
    f.close()
if args.verbose:
    print('Done!')
