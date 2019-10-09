import argparse
import json
import pickle
import warnings

from grr.Filter_Exps import Filter_Exps
from grr.Tools import gagProcess
# Model to be fitted is imported lower down.


# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument('experiments', help='Path to pickled experiments.')
parser.add_argument('output', help='File to save fitted models.')
parser.add_argument('opts', help='JSON file with fitting options.')
parser.add_argument(
    '-m', '--model', type=str, default='GIF',
    help='Kind of model to fit to data (case-insensitive). Default GIF.'
)
parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Print information about progress.'
)

# Parse commandline args.
args = parser.parse_args()

# Raise an error if invalid model type is provided.
valid_models = ['gif', 'augmentedgif', 'igif_np', 'igif_vr']
if args.model.lower() not in valid_models:
    raise ValueError(
        'Invalid argument model={}. Expected one of {}.'.format(
            args.model, valid_models
        )
    )

# Load opts from JSON file.
with open(args.opts, 'r') as f:
    opts = json.load(f)
    f.close()

# Check that JSON opts file has correct fields.
# Fields required for all models.
required_json_fields = ['T_ref', 'eta_timescales', 'gamma_timescales',
                        'DT_beforeSpike', 'dt']
# Add model-specific fields.
if args.model.lower() == 'gif':
    pass
elif args.model.lower() == 'augmentedgif':
    required_json_fields.append('AugmentedGIF_fit_args')
elif args.model.lower() == 'igif_np':
    required_json_fields.append('iGIF_NP_fit_args')
elif args.model.lower() == 'igif_vr':
    required_json_fields.append('iGIF_VR_fit_args')
else:
    warnings.warn(
        'Required opts JSON fields for model `{}` are undefined and could be '
        'missing.'.format(args.model)
    )
# Run check. (Does not check model-specific subfields.)
for field in required_json_fields:
    if field not in opts:
        raise AttributeError(
            '{optsfile} missing required field {fieldname}'.format(
                optsfile=args.opts, fieldname=field
            )
        )

# LOAD EXPERIMENTS

if args.verbose:
    print('Loading experiments from {}'.format(args.experiments))
with open(args.experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()
if args.verbose:
    print('Done!')


# FIT MODELS

# Load specified model.
if args.model.lower() == 'gif':
    from grr.GIF import GIF
    model_type = GIF
elif args.model.lower() == 'augmentedgif':
    from grr.AugmentedGIF import AugmentedGIF
    model_type = AugmentedGIF
elif args.model.lower() == 'igif_np':
    from grr.iGIF import iGIF_NP
    model_type = iGIF_NP
elif args.model.lower() == 'igif_vr':
    from grr.iGIF import iGIF_VR
    model_type = iGIF_VR
else:
    raise RuntimeError('Could not load model {}'.format(args.model))

# Fit model to each experiment.
models = []
for i, expt in enumerate(experiments):

    if args.verbose:
        print(
            '\nFitting {} to {} ({:.1f}%)'.format(
                args.model,
                expt.name,
                100 * (i + 1) / len(experiments)
            )
        )

    tmp_mod = model_type(opts['dt'])
    tmp_mod.name = expt.name

    # Set hyperparameters.
    tmp_mod.Tref = opts['T_ref']
    tmp_mod.eta = Filter_Exps()
    tmp_mod.eta.setFilter_Timescales(opts['eta_timescales'])
    tmp_mod.gamma = Filter_Exps()
    tmp_mod.gamma.setFilter_Timescales(opts['gamma_timescales'])

    # Fit model silently.
    with gagProcess():
        if args.model.lower() == 'gif':
            tmp_mod.fit(
                expt,
                DT_beforeSpike=opts['DT_beforeSpike']
            )
        elif args.model.lower() == 'augmentedgif':
            tmp_mod.fit(
                expt,
                DT_beforeSpike=opts['DT_beforeSpike'],
                **opts['AugmentedGIF_fit_args']
            )
        elif args.model.lower() == 'igif_np':
            tmp_mod.fit(
                expt,
                DT_beforeSpike=opts['DT_beforeSpike'],
                **opts['iGIF_NP_fit_args']
            )
        elif args.model.lower() == 'igif_vr':
            tmp_mod.fit(
                expt,
                DT_beforeSpike=opts['DT_beforeSpike'],
                **opts['iGIF_VR_fit_args']
            )
        else:
            raise RuntimeError('Could not fit model {}'.format(args.model))

    # Clear interpolated filter cache to save space when saved.
    tmp_mod.eta.clearInterpolatedFilter()
    tmp_mod.gamma.clearInterpolatedFilter()

    models.append(tmp_mod)

    if args.verbose:
        tmp_mod.printParameters()

if args.verbose:
    print('Done fitting {}s!'.format(args.model))


# SAVE MODELS TO DISK

if args.verbose:
    print('Saving fitted models to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(models, f)
if args.verbose:
    print('Finished! Exiting.')
