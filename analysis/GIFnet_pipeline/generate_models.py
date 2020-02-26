#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse
import json

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


# HELPER FUNCTION

def construct_file_name(number, kind):
    """Get file name for saving gifnet.

    Filename format is:
        `<args.prefix>_<number>_<kindsuffix>`

    """
    if kind not in opts['output_model_suffixes'].keys():
        raise ValueError('Unrecognized model kind {}'.format(kind))
    fname = '_'.join(
        [args.prefix, str(number), opts['output_model_suffixes'][kind]]
    )
    return fname


# GENERATE MODELS

gaba_kernel = BiexponentialSynapticKernel(
    size=opts['gaba_input']['amplitude'],
    tau_rise=opts['gaba_input']['tau_rise'],
    tau_decay=opts['gaba_input']['tau_decay'],
    size_method='amplitude',
    duration=opts['gaba_input']['duration'],
    dt=opts['dt'],
    front_padded=True
)

subsample_builder = gfn.SubsampleGIFnetBuilder(
    sergifs,
    somgifs,
    opts['no_ser_neurons'],
    opts['no_gaba_neurons'],
    opts['propagation_delay'],
    gaba_kernel,
    opts['gaba_input']['reversal'],
    opts['connection_probability'],
    opts['dt'],
    'base',
)
homogenous_builder = gfn.HomogenousGIFnetBuilder(
    sergifs,
    somgifs,
    opts['no_ser_neurons'],
    opts['no_gaba_neurons'],
    opts['propagation_delay'],
    gaba_kernel,
    opts['gaba_input']['reversal'],
    opts['connection_probability'],
    opts['dt'],
    'homogenous',
)

for i in range(args.replicates):
    if args.verbose:
        print('Assembling GIFnet model set {} of {}.'.format(
            i+1, args.replicates
        ))

    # Vanilla model.
    subsample_builder.random_build()
    subsample_builder.export_to_file(construct_file_name(i, 'base'))

    # Fixed IA.
    fixedIA_builder = gfn.FixedIAGIFnetBuilder(subsample_builder, opts['dt'], 'fixedIA')
    fixedIA_builder.fix_IA(opts['fixed_IA_conductance'], None)
    fixedIA_builder.export_to_file(construct_file_name(i, 'fixedIA'))

    # IA knockout.
    fixedIA_builder.label = 'noIA'
    fixedIA_builder.fix_IA(0., None)
    fixedIA_builder.export_to_file(construct_file_name(i, 'noIA'))

    # Model with swapped adaptation mechanisms.
    swapped_adaptation_builder = gfn.SwappedAdaptationGIFnetBuilder(subsample_builder, opts['dt'], 'adaptation_swap')
    swapped_adaptation_builder.swap_adaptation()
    swapped_adaptation_builder.export_to_file(construct_file_name(i, 'adaptation_swap'))

    # Model with homogenous 5HT and GABA.
    homogenous_builder.homogenous_build(homogenous_5HT=True, homogenous_GABA=True)
    homogenous_builder.export_to_file(construct_file_name(i, 'homogenous'))

    # Model with homogenous 5HT and GABA and swapped adaptation.
    homogenous_swapped_builder = gfn.SwappedAdaptationGIFnetBuilder(homogenous_builder, opts['dt'], 'homogenous_adaptation_swap')
    homogenous_swapped_builder.swap_adaptation()
    homogenous_builder.export_to_file(construct_file_name(i, 'homogenous_adaptation_swap'))

    # Model with homogenous GABA and heterogenous 5HT.
    homogenous_builder.homogenous_build(homogenous_5HT=False, homogenous_GABA=True)
    homogenous_builder.export_to_file(construct_file_name(i, 'homogenous_GABA_only'))


if args.verbose:
    print('Finished! Exiting.')
