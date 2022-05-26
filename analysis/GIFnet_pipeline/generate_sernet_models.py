import pickle
from copy import deepcopy

import numpy as np

import grr.GIF_network as gfn

from lib import generate_models_argparser, load_generate_models_opts

# Parse commandline arguments and externally-configured options.
args = generate_models_argparser.parse_args()
opts = load_generate_models_opts(
    args.opts,
    {
        'dt': None,
        'no_principal_neurons': None,
        'potassium_reversal_potential': None,
    },
)

# Load GIF models
if args.verbose:
    print('Loading 5HT models from {}'.format(args.mods))
with open(args.mods, 'rb') as f:
    principal_cell_gifs = pickle.load(f)
    f.close()

# Set potassium reversal potential.
for model in principal_cell_gifs:
    model.E_K = float(opts['potassium_reversal_potential'])

# Set random seed.
np.random.seed(args.seed)

# Generate models
for i in range(args.replicates):
    if args.verbose:
        print(
            'Assembling GIFnet model set {} of {}.'.format(
                i + 1, args.replicates
            )
        )

    subsample_gifnet = gfn.GIFnet(
        name='Subsample GIFs',
        ser_mod=np.random.choice(
            deepcopy(principal_cell_gifs), opts['no_principal_neurons']
        ),
        gaba_mod=None,
        propagation_delay=None,
        gaba_kernel=None,
        gaba_reversal=None,
        connectivity_matrix=[],
        dt=opts['dt'],
    )

    # Clear cached interpolated filters to save disk space.
    subsample_gifnet.clear_interpolated_filters()

    # Save base model to disk.
    fname = '_'.join(
        [args.prefix, str(i), 'subsample_base.mod']
    )
    if args.verbose:
        print('Saving {} GIFnet model to {}'.format(type, fname))
    with open(fname, 'wb') as f:
        pickle.dump(subsample_gifnet, f)
        f.close()

if args.verbose:
    print('Finished! Exiting.')
