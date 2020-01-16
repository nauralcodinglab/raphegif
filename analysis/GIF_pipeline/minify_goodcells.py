import argparse
import pickle
import json
import warnings

from grr.Minify import (
    ExperimentMinifier,
    SpikeCount_TraceMinifier,
    InsufficientSpikesError,
)
from grr.Tools import check_dict_fields, reprint

# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'experiments', type=str,
    help='Path to pickled list of experiments'
)
parser.add_argument('opts', type=str)
parser.add_argument('output', type=str, help='Path to output.')
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print('Loading opts from {}'.format(args.opts))
with open(args.opts, 'r') as f:
    opts = json.load(f)
    f.close()

opts_template = {
    'train': {'num_spikes': None, 'pad': None, 'mode': None},
    'test': {'num_spikes': None, 'pad': None, 'mode': None},
}
check_dict_fields(opts, opts_template, raise_error=True)


# LOAD PROCESSED EXPERIMENTS

if args.verbose:
    print('Loading experiments from {}'.format(args.experiments))
with open(args.experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()


# MINIFY EXPERIMENTS

expt_minifier = ExperimentMinifier(
    SpikeCount_TraceMinifier(
        opts['train']['num_spikes'],
        opts['train']['pad'],
        opts['train']['mode']
    ),
    SpikeCount_TraceMinifier(
        opts['test']['num_spikes'],
        opts['test']['pad'],
        opts['test']['mode']
    )
)

minified_experiments = []
insufficient_spikes_labels = []
for i, expt in enumerate(experiments):
    if args.verbose:
        reprint(
            "Minifying experiments {:.1f}%".format(
                100.0 * i / len(experiments)
            )
        )
    try:
        minified_experiments.append(expt_minifier.minify(expt))
    except InsufficientSpikesError:
        insufficient_spikes_labels.append(expt.name)

if len(insufficient_spikes_labels) > 0:
    warnings.warn(
        "Excluding {} experiments without required {} training and {} test"
        " spikes/sweep:\n {}".format(
            len(insufficient_spikes_labels),
            expt_minifier.trainingsetMinifier.numberOfSpikes,
            expt_minifier.testsetMinifier.numberOfSpikes,
            insufficient_spikes_labels,
        )
    )
if args.verbose:
    print("\nDone!")


# SAVE MINIFIED EXPERIMENTS

if args.verbose:
    print('Saving minified experiments to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(minified_experiments, f)
if args.verbose:
    print('Finished! Exiting.')
