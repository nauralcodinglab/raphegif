#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse
import json
from copy import deepcopy

import numpy as np

import grr.GIF_network as gfn
from grr.Tools import check_dict_fields
from ezephys.stimtools import BiexponentialSynapticKernel


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    '--sermods', type=str, required=True,
    help='Pickled serotonin neuron models.'
)
parser.add_argument(
    '--gabamods', type=str, required=True,
    help='Pickled GABA neuron models.'
)
parser.add_argument(
    '--prefix', type=str, required=True,
    help='Path to save GIF_network models.'
)
parser.add_argument(
    '--opts', type=str, required=True,
    help='Path to opts JSON file.'
)
parser.add_argument(
    '-r', '--replicates', default=1, type=int,
    help='No. of randomized models to generate.'
)
parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed (default 42).'
)
parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Print information about progress.'
)

args = parser.parse_args()

# Parse JSON opts file.
with open(args.opts, 'r') as f:
    opts = json.load(f)
    f.close()
# Ensure JSON object contains required fields.
required_fields = {
    'dt': None,
    'propagation_delay': None,
    'gaba_input': {
        'tau_rise': None,
        'tau_decay': None,
        'amplitude': None,
        'reversal': None,
        'duration': None
    },
    'no_ser_neurons': None,
    'no_gaba_neurons': None,
    'connection_probability': None,
    'fixed_IA_conductance': None,
    'output_model_suffixes': {
        'base': None,
        'noIA': None,
        'fixedIA': None,
        'adaptation_swap': None,
        'homogenous_adaptation_swap': None,
        'homogenous': None,
        'homogenous_GABA_only': None
    }
}
check_dict_fields(opts, required_fields)


#%% LOAD GIF MODELS

if args.verbose:
    print('Loading 5HT models from {}'.format(args.sermods))
with open(args.sermods, 'rb') as f:
    sergifs = pickle.load(f)
    f.close()

if args.verbose:
    print('Loading GABA models from {}'.format(args.gabamods))
with open(args.gabamods, 'rb') as f:
    somgifs = pickle.load(f)
    f.close()
if args.verbose:
    print('Done loading single cell models!')


# SET RANDOM SEED

np.random.seed(args.seed)


# GENERATE MODELS

def save_model(model, type, number):
    """Save GIFnet model to a pickle file.

    `model` is saved to
        `<prefix>_<number>_<typesuffix>`
    according to args and opts.

    Arguments
    ---------
    model : GIFnet
        Model to save.
    type : str in {`base`, `noIA`, `fixedIA`}
        Type of model being saved. Used to get correct filename suffix.
    number : int
        Number of model being saved. Used as filename infix.

    """
    fname = '_'.join(
        [args.prefix, str(number), opts['output_model_suffixes'][type]]
    )
    if args.verbose:
        print('Saving {} GIFnet model to {}'.format(type, fname))
    with open(fname, 'wb') as f:
        pickle.dump(model, f)
        f.close()

def set_5HT_IA_in_gifnet(gifnet, g_A):
    """Set IA max conductance for all 5HT cells in copy of gifnet."""
    gifnet = deepcopy(gifnet)
    for ind in range(len(gifnet.ser_mod)):
        gifnet.ser_mod[ind].gbar_K1 = g_A
    return gifnet


def generate_swapped_adaptation_gifnet(gifnet, ser_adaptation_donors, gaba_adaptation_donors):
    """Generate new gifnet by grafting adaptation kernels onto opposite cell type."""
    gifnet = deepcopy(gifnet)

    adaptation_donors = np.random.choice(gaba_adaptation_donors, len(gifnet.ser_mod))
    for ind in range(len(gifnet.ser_mod)):
        gifnet.ser_mod[ind].eta.setFilter_Coefficients(
            adaptation_donors[ind].eta.getCoefficients()
        )
        gifnet.ser_mod[ind].gamma.setFilter_Coefficients(
            adaptation_donors[ind].gamma.getCoefficients()
        )

    adaptation_donors = np.random.choice(ser_adaptation_donors, len(gifnet.gaba_mod))
    for ind in range(len(gifnet.gaba_mod)):
        gifnet.gaba_mod[ind].eta.setFilter_Coefficients(
            adaptation_donors[ind].eta.getCoefficients()
        )
        gifnet.gaba_mod[ind].gamma.setFilter_Coefficients(
            adaptation_donors[ind].gamma.getCoefficients()
        )

    return gifnet


def generate_random_connectivity_matrix():
    connectivity_matrix = (
        np.random.uniform(
            size=(opts['no_ser_neurons'], opts['no_gaba_neurons'])
        ) < opts['connection_probability']
    ).astype(np.int8)
    return connectivity_matrix




gaba_kernel = BiexponentialSynapticKernel(
    amplitude=opts['gaba_input']['amplitude'],
    tau_rise=opts['gaba_input']['tau_rise'],
    tau_decay=opts['gaba_input']['tau_decay'],
    duration=opts['gaba_input']['duration'],
    dt=opts['dt'],
    front_padded=True
)

for i in range(args.replicates):
    if args.verbose:
        print('Assembling GIFnet model set {} of {}.'.format(
            i+1, args.replicates
        ))

    connectivity_matrix = (
        np.random.uniform(
            size=(opts['no_ser_neurons'], opts['no_gaba_neurons'])
        ) < opts['connection_probability']
    ).astype(np.int8)

    subsample_gifnet = gfn.GIFnet(
        name='Subsample GIFs',
        ser_mod=np.random.choice(deepcopy(sergifs), opts['no_ser_neurons']),
        gaba_mod=np.random.choice(deepcopy(somgifs), opts['no_gaba_neurons']),
        propagation_delay=opts['propagation_delay'],
        gaba_kernel=gaba_kernel.kernel,
        gaba_reversal=opts['gaba_input']['reversal'],
        connectivity_matrix=connectivity_matrix,
        dt=opts['dt']
    )
    del connectivity_matrix

    # Clear cached interpolated filters to save disk space.
    subsample_gifnet.clear_interpolated_filters()

    # Save base model to disk.
    save_model(subsample_gifnet, 'base', i)

    # Model with fixed IA conductance.
    fixed_IA_gifnet = set_5HT_IA_in_gifnet(
        subsample_gifnet, opts['fixed_IA_conductance']
    )
    save_model(fixed_IA_gifnet, 'fixedIA', i)

    # Model with IA knocked out.
    IA_KO_gifnet = set_5HT_IA_in_gifnet(subsample_gifnet, 0.)
    save_model(IA_KO_gifnet, 'noIA', i)

    # Model with swapped adaptation mechanisms.
    swapped_adaptation_gifnet = generate_swapped_adaptation_gifnet(
        subsample_gifnet, sergifs, somgifs
    )
    save_model(swapped_adaptation_gifnet, 'adaptation_swap', i)


if args.verbose:
    print('Finished! Exiting.')
