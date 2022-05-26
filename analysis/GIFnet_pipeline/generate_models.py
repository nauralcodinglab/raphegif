#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse
import json
from os.path import exists

import numpy as np

import grr.GIF_network as gfn
from grr.Tools import check_dict_fields
from ezephys.stimtools import BiexponentialSynapticKernel

from lib import generate_models_argparser


#%% PARSE COMMANDLINE ARGUMENTS

args = generate_models_argparser.parse_args()

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
        'duration': None,
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
        'homogenous_GABA_only': None,
    },
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


def safe_export(builder, number, model_kind):
    """Only export if the model doesn't already exist, or overwrite is set.

    Parameters
    ----------
    builder : GIFnetBuilder
    model_kind : str
        Used for the file name.

    """
    file_name = construct_file_name(number, model_kind)
    if args.overwrite:
        if exists(file_name):
            print('Model {} already exists. Overwriting.'.format(file_name))
        builder.export_to_file(file_name)
    elif not exists(file_name):
        builder.export_to_file(file_name)
    elif args.verbose:
        print('Model {} already exists. Skipping.'.format(file_name))


# GENERATE MODELS

gaba_kernel = BiexponentialSynapticKernel(
    size=opts['gaba_input']['amplitude'],
    tau_rise=opts['gaba_input']['tau_rise'],
    tau_decay=opts['gaba_input']['tau_decay'],
    size_method='amplitude',
    duration=opts['gaba_input']['duration'],
    dt=opts['dt'],
    front_padded=True,
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
        print(
            'Assembling GIFnet model set {} of {}.'.format(
                i + 1, args.replicates
            )
        )

    # Vanilla model.
    subsample_builder.random_build()
    safe_export(subsample_builder, i, 'base')

    # Model with 5HT DV replaced by GABA value
    swapped_dv_builder = gfn.SwappedDVGIFnetBuilder(
        subsample_builder, opts['dt'], 'dv_swap_ser_only'
    )
    swapped_dv_builder.graft_gaba_dv_onto_ser()
    safe_export(swapped_dv_builder, i, 'dv_swap_ser_only')

    # Fixed IA.
    fixedIA_builder = gfn.FixedIAGIFnetBuilder(
        subsample_builder, opts['dt'], 'fixedIA'
    )
    fixedIA_builder.fix_IA(opts['fixed_IA_conductance'], None)
    safe_export(fixedIA_builder, i, 'fixedIA')

    # IA knockout.
    fixedIA_builder.label = 'noIA'
    fixedIA_builder.fix_IA(0.0, None)
    safe_export(fixedIA_builder, i, 'noIA')

    # Model with 5HT adaptation replaced by GABA adaptation.
    swapped_adaptation_builder = gfn.SwappedAdaptationGIFnetBuilder(
        subsample_builder, opts['dt'], 'adaptation_swap_ser_only'
    )
    swapped_adaptation_builder.swap_adaptation(
        gaba_onto_ser=True, ser_onto_gaba=False
    )
    safe_export(swapped_adaptation_builder, i, 'adaptation_swap_ser_only')

    # Model with 5HT adaptation AND DV replaced by GABA values
    swapped_dv_adaptation_builder = gfn.SwappedDVGIFnetBuilder(
        swapped_adaptation_builder, opts['dt'], 'dv_adaptation_swap_ser_only'
    )
    swapped_dv_adaptation_builder.graft_gaba_dv_onto_ser()
    safe_export(
        swapped_dv_adaptation_builder, i, 'dv_adaptation_swap_ser_only'
    )

    # Model with 5HT adaptation replaced by GABA AND VICE VERSA
    swapped_adaptation_builder.swap_adaptation()
    safe_export(swapped_adaptation_builder, i, 'adaptation_swap')

    # Model with homogenous 5HT and GABA.
    homogenous_builder.homogenous_build(
        homogenous_5HT=True, homogenous_GABA=True
    )
    safe_export(homogenous_builder, i, 'homogenous')

    # Model with homogenous 5HT and GABA and swapped adaptation.
    homogenous_swapped_builder = gfn.SwappedAdaptationGIFnetBuilder(
        homogenous_builder, opts['dt'], 'homogenous_adaptation_swap'
    )
    homogenous_swapped_builder.swap_adaptation()
    safe_export(homogenous_builder, i, 'homogenous_adaptation_swap')

    # Model with homogenous GABA and heterogenous 5HT.
    homogenous_builder.homogenous_build(
        homogenous_5HT=False, homogenous_GABA=True
    )
    safe_export(homogenous_builder, i, 'homogenous_GABA_only')


if args.verbose:
    print('Finished! Exiting.')
