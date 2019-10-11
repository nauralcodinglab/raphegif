import os
import pickle
import argparse
import warnings

import pandas as pd

from grr.Tools import gagProcess


# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'experiments', type=str,
    help='Pickled list of Experiment objects'
)
parser.add_argument(
    'models', type=str, nargs='+',
    help='Pickled lists of GIF or GIF-like models to use to compute '
    'spiketrain predictions.'
)
parser.add_argument(
    '-o', '--output', type=str, required=True,
    help='Directory in which to save output. Filenames are generated '
    'automatically.'
)
parser.add_argument(
    '-v', '--verbose', help='Print information about progress.',
    action='store_true'
)
parser.add_argument(
    '--precision', help='Width of window to consider coincidence (ms).',
    default=4., type=float
)
parser.add_argument(
    '--repeats', help='Number of spiketrains to realize from model.',
    default=500, type=int
)
args = parser.parse_args()

# Check arguments.
if not all([os.path.exists(modfname) for modfname in args.models]):
    raise ValueError(
        'In argument `models`: file(s) {} not found.'.format(
            [modfname for modfname in args.models if not os.path.exists(modfname)]
        )
    )
if not os.path.isdir(args.output):
    raise ValueError(
        'Invalid value for argument `--output`: {} is not a directory.'.format(
            args.output
        )
    )


# LOAD MODELS/DATA

if args.verbose:
    print('Loading experiments from {}'.format(args.experiments))
with open(args.experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()

if args.verbose:
    print('Loading models from {}'.format(args.models))
models = {}
for modfname in args.models:
    with open(modfname, 'rb') as f:
        label = os.path.basename(modfname).split('.')[0]
        models[label] = pickle.load(f)
        f.close()

# Check that there are at least as many expts as each kind of model.
if any([len(experiments) > len(models[label]) for label in models]):
    warnings.warn('More experiments than {} models.'.format(
        [label for label in models if len(experiments) > len(models[label])]
    ))
# If there are fewer experiments than models, something is very wrong.
if any([len(experiments) < len(models[label]) for label in models]):
    raise RuntimeError(
        'Fewer experiments than {} models. Not all models can be '
        'benchmarked.'.format(label)
    )


# RUN BENCHMARKS

# Initialize tables to hold 3 benchmarks: sample traces, Md*, and subthreshold R^2.
benchmarks = {}
for benchmark in ['sample_traces', 'Md_vals', 'R2_V_vals', 'R2_dV_vals']:
    benchmarks[benchmark] = {label: [] for label in models}
    benchmarks[benchmark]['Cell'] = [expt.name for expt in experiments]  # Column for cell identifier.
benchmarks['sample_traces']['Time'] = []  # Extra column to hold time support vectors.
benchmarks['sample_traces']['Input'] = []  # Extra column for model/neuron input.
benchmarks['sample_traces']['Data'] = []  # Extra column for recorded V/spikes.

# For each experiment and model type compute benchmarks.
for i, expt in enumerate(experiments):
    if args.verbose:
        print('\nRunning benchmarks for cell {} ({}%)'.format(
            expt.name, 100. * i / len(experiments)
        ))

    # Store sample trace from raw data.
    benchmarks['sample_traces']['Time'].append(
        expt.testset_traces[0].getTime()
    )
    benchmarks['sample_traces']['Input'].append(expt.testset_traces[0].I)
    benchmarks['sample_traces']['Data'].append({
        'V': expt.testset_traces[0].V,
        'spks': [tr.getSpikeTimes() for tr in expt.testset_traces]
    })

    # Benchmark each model.
    for label in models:

        # Select model matching cell identifier from expt.
        found_matching_mod = False
        for mod in models[label]:
            if mod.name == expt.name:
                found_matching_mod = True
                break
            else:
                continue
        # Store NaNs if no match found.
        if not found_matching_mod:
            warnings.warn(
                'No {label} model matching {name} found. Using `None` for '
                'benchmarks.'.format(label=label, name=expt.name)
            )
            for benchmark in benchmarks:
                benchmarks[benchmark][label].append(None)
            continue

        assert mod.name == expt.name, '{} model {} does not match experiment {}.'.format(label, mod.name, expt.name)

        # Benchmark 1: sample trace
        if args.verbose:
            print(
                'Getting sample trace for {} model {}'.format(label, mod.name)
            )
        tmp_spks = []
        for j in range(len(expt.testset_traces)):
            time, V, _, _, spks = mod.simulate(
                expt.testset_traces[0].I,
                expt.testset_traces[0].V[0]
            )
            tmp_spks.append(spks)
        benchmarks['sample_traces'][label].append({
            'V': V,
            'spks': tmp_spks
        })
        del time, V, _, tmp_spks

        # Benchmark 2: Md*
        if args.verbose:
            print(
                'Getting Md* ({:.1f}ms precision) for {} model {}'.format(
                    args.precision, label, mod.name
                )
            )
        with gagProcess():
            tmp_prediction = expt.predictSpikes(mod, nb_rep=args.repeats)
            benchmarks['Md_vals'][label].append(
                tmp_prediction.computeMD_Kistler(args.precision, expt.dt)
            )
        del tmp_prediction

        # Benchmark 3: subthreshold R^2 on V.
        if args.verbose:
            print(
                'Getting subthreshold R^2 on V for {} model {}'.format(
                    label, mod.name
                )
            )
        benchmarks['R2_V_vals'][label].append(mod.var_explained_V)

        # Benchmark 4: subthreshold R^2 on dV.
        if args.verbose:
            print(
                'Getting subthreshold R^2 on dV for {} model {}'.format(
                    label, mod.name
                )
            )
        benchmarks['R2_dV_vals'][label].append(mod.var_explained_dV)

        # Cleanup.
        del mod

    if args.verbose:
        print('Done cell {}.'.format(expt.name))

print('\nDone benchmarks!')


# SAVE OUTPUT

# Save sample traces.
sample_tr_fname = (
    os.path.basename(args.experiments).split('.')[0]
    + 'benchmark_sample_traces.pkl'
)
if args.verbose:
    print('Saving sample traces to {}'.format(
        os.path.join(args.output, sample_tr_fname)
    ))
with open(os.path.join(args.output, sample_tr_fname), 'wb') as f:
    pickle.dump(benchmarks['sample_traces'], f)
    f.close()
del sample_tr_fname

# Save Md*.
Md_fname = (
    os.path.basename(args.experiments).split('.')[0]
    + 'benchmark_Md_{:d}.csv'.format(int(args.precision))
)
if args.verbose:
    print('Saving Md* to {}'.format(os.path.join(args.output, Md_fname)))
pd.DataFrame(benchmarks['Md_vals']).to_csv(Md_fname, index=False)

# Save R^2 on V.
R2_V_fname = (
    os.path.basename(args.experiments).split('.')[0]
    + 'benchmark_R2_V.csv'
)
if args.verbose:
    print(
        'Saving R^2 on subthreshold V to {}'.format(
            os.path.join(args.output, R2_V_fname)
        )
    )
pd.DataFrame(benchmarks['R2_V_vals']).to_csv(R2_V_fname, index=False)

# Save R^2 on V.
R2_dV_fname = (
    os.path.basename(args.experiments).split('.')[0]
    + 'benchmark_R2_dV.csv'
)
if args.verbose:
    print(
        'Saving R^2 on subthreshold dV to {}'.format(
            os.path.join(args.output, R2_dV_fname)
        )
    )
pd.DataFrame(benchmarks['R2_dV_vals']).to_csv(R2_dV_fname, index=False)

# Exit.
if args.verbose:
    print('\nFinished! Exiting.')
