from __future__ import division

import pickle
import argparse
import json
from copy import deepcopy

import numpy as np

from grr.Tools import (
    check_dict_fields,
    validate_matching_axis_lengths,
    timeToIndex,
)


# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument('params', type=str, help='JSON file with parameters.')
parser.add_argument(
    'output',
    type=str,
    help='Path to save output. '
    'Pickled dict with ser_input and gaba_input as attributes.',
)
parser.add_argument(
    '-v',
    '--verbose',
    help='Print more information about progress.',
    action='store_true',
)

args = parser.parse_args()

if args.verbose:
    print('Loading params from {}'.format(args.params))
with open(args.params, 'r') as f:
    params = json.load(f)
    f.close()

# Ensure `params` has correct fields.
params_template = {
    "dt": None,
    "duration": None,
    "ser": {
        "start": None,
        "duration": None,
        "baseline": None,
        "amplitude": None
    },
    "gaba": {
        "start": None,
        "duration": None,
        "baseline": None,
        "amplitude": None
    }
}
check_dict_fields(params, params_template, raise_error=True)
validate_matching_axis_lengths(
    [
        params['ser']['baseline'],
        params['ser']['amplitude'],
        params['gaba']['baseline'],
        params['gaba']['amplitude'],
    ],
    [0],
)
no_sweeps = len(params['ser']['baseline'])

# GENERATE IMPULSE
def generate_input_basis(
    baseline_duration, pulse_duration, total_duration, dt
):
    baseline_int = timeToIndex(baseline_duration, dt)[0]
    pulse_int = timeToIndex(pulse_duration, dt)[0]
    tail_int = timeToIndex(
        total_duration - baseline_duration - pulse_duration, dt
    )[0]
    scalable_arr = np.concatenate(
        (
            np.zeros((no_sweeps, 1, baseline_int)),
            np.ones((no_sweeps, 1, pulse_int)),
            np.zeros((no_sweeps, 1, tail_int)),
        ),
        axis=2,
    )
    return scalable_arr


def rescale_and_offset_input_basis(basis, scale, offset):
    assert np.ndim(basis) == 3, 'Input basis must be an array with dimensionality [sweeps, neurons, timesteps]'

    # Reshape scaling parameters for numpy broadcasting.
    scale_column = np.atleast_1d(scale)[:, np.newaxis, np.newaxis]
    offset_column = np.atleast_1d(offset)[:, np.newaxis, np.newaxis]

    # Rescale and offset input array.
    input_arr = deepcopy(basis)
    input_arr *= scale_column
    input_arr += offset_column

    return input_arr


ser_input = generate_input_basis(
    params['ser']['start'],
    params['ser']['duration'],
    params['duration'],
    params['dt'],
)
ser_input = rescale_and_offset_input_basis(
    ser_input, params['ser']['amplitude'], params['ser']['baseline']
)

gaba_input = generate_input_basis(
    params['gaba']['start'],
    params['gaba']['duration'],
    params['duration'],
    params['dt'],
)
gaba_input = rescale_and_offset_input_basis(
    gaba_input, params['gaba']['amplitude'], params['gaba']['baseline']
)

# SAVE NETWORK INPUT
network_input = {
    'metaparams': {
        'input_type': 'impulse',
        'sweeps': no_sweeps,
        'duration': params['duration'],
        'dt': params['dt'],
        'ser_baseline': params['ser']['baseline'],
        'ser_time': params['ser']['start'],
        'ser_amplitude': params['ser']['amplitude'],
        'gaba_baseline': params['gaba']['baseline'],
        'gaba_time': params['gaba']['start'],
        'gaba_amplitude': params['gaba']['amplitude'],
    },
    'ser_input': ser_input,
    'gaba_input': gaba_input
}

if args.verbose:
    print('Saving impulse to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(network_input, f)
    f.close()
if args.verbose:
    print('Finished! Exiting.')
