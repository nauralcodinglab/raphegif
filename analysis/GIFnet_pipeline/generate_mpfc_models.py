import pickle
import argparse
import json
from copy import deepcopy

import numpy as np

import grr.GIF_network as gfn
from grr.Tools import check_dict_fields


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mods', type=str, required=True,
    help='Pickled neuron models.'
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
    'no_principal_neurons': None,
    'output_model_suffixes': {'base': None, 'noIA': None, 'fixedIA': None}
}
check_dict_fields(opts, required_fields)


#%% LOAD GIF MODELS

if args.verbose:
    print('Loading mPFC models from {}'.format(args.mods))
with open(args.mods, 'rb') as f:
    principal_cell_gifs = pickle.load(f)
    f.close()


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


for i in range(args.replicates):
    if args.verbose:
        print('Assembling GIFnet model set {} of {}.'.format(
            i+1, args.replicates
        ))

    subsample_gifnet = gfn.GIFnet(
        name='Subsample GIFs',
        ser_mod=np.random.choice(deepcopy(principal_cell_gifs), opts['no_principal_neurons']),
        gaba_mod=None,
        propagation_delay=None,
        gaba_kernel=None,
        gaba_reversal=None,
        connectivity_matrix=[],
        dt=opts['dt']
    )

    # Clear cached interpolated filters to save disk space.
    subsample_gifnet.clear_interpolated_filters()

    # Save base model to disk.
    save_model(subsample_gifnet, 'base', i)

if args.verbose:
    print('Finished! Exiting.')
