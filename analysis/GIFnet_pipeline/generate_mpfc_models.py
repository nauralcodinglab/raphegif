import pickle
from copy import deepcopy

import numpy as np

import grr.GIF_network as gfn

from lib import generate_models_argparser, load_generate_models_opts


# Parse commandline arguments
args = generate_models_argparser.parse_args()
opts = load_generate_models_opts(
    args.opts,
    {
        'dt': None,
        'no_principal_neurons': None,
        'output_model_suffixes': {'base': None, 'noIA': None, 'fixedIA': None},
    },
)

# Load single neuron models.
if args.verbose:
    print('Loading mPFC models from {}'.format(args.mods))
with open(args.mods, 'rb') as f:
    principal_cell_gifs = pickle.load(f)
    f.close()

# Set random seed.
np.random.seed(args.seed)

# Generate models.
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
    save_model(subsample_gifnet, 'base', i)

if args.verbose:
    print('Finished! Exiting.')
