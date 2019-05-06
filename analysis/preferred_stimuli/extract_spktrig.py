"""Command line script to extract spike-triggered stimulus and voltage from preprocessed Experiments.
"""

#%% IMPORT MODULES

from __future__ import division

import pickle
import argparse
import warnings

import numpy as np


#%% PARSE COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'Experiments', help = 'Path to pickled list of Experiments from which to extract spike-triggered input slices.'
)
parser.add_argument(
    'output', help = 'Path to output processed data.'
)
parser.add_argument(
    '--before', help = 'Start of time interval around spikes to extract input (ms).',
    type = float, default = 200.
)
parser.add_argument(
    '--after', help = 'End of time interval around spikes to extract input (ms).',
    type = float, default = 100.
)
parser.add_argument(
    '-v', '--verbose', help = 'Print more progress messages.',
    action = 'store_true'
)
args = parser.parse_args()


#%% LOAD EXPERIMENTS AND PERFORM CHECKS

if args.verbose:
    print 'Loading Experiments from {}'.format(args.Experiments)
with open(args.Experiments, 'rb') as f:
    experiments = pickle.load(f)
    f.close()

# Check trace channels for sampling rates and lengths.
if args.verbose:
    print 'Running input checks.'

dts = []
I_lens = []
V_lens = []
for expt in experiments:
    for tr in expt.trainingset_traces:
        dts.append(tr.dt)
        I_lens.append(len(tr.I))
        V_lens.append(len(tr.V))

if not np.allclose(dts, dts[0]):
    warnings.warn(
        'Sampling rate not equal for all trainingset_traces. '
        'Continuing, but this could cause problems in downstream analysis.',
        RuntimeWarning
    )
del dts

if not np.allclose(I_lens, V_lens):
    raise ValueError(
        'Sampling rate differs between I and V channels in some traces.'
    )
del I_lens, V_lens


#%% EXTRACT INPUT AROUND SPIKES

spike_triggered_slices = {
    'I': [],
    'V': [],
    't': [],
    't_spk': [],
    't_since_spk': [],
    'names': []
}

for i, expt in enumerate(experiments):

    if args.verbose:
        print 'Extracting spikes from {} ({:.1f}%)'.format(
            expt.name, 100 * (i + 1) / len(experiments)
        )

    for tr in expt.trainingset_traces:

        # Handle case that spikes have not yet been detected.
        if not tr.spks_flag:
            tr.detectSpikes()

        # Compute width of interval in TIMESTEPS.
        int_DT_beforespk = int(round(args.before/tr.dt))
        int_DT_afterspk = int(round(args.after/tr.dt))

        # Allocate temporary matrices to hold output.
        # (Will be converted to lists of lists, but this allocates memory in one shot.)
        spk_trig_I = np.full(
            (len(tr.spks), int_DT_beforespk + int_DT_afterspk),
            np.nan,
            dtype = np.float32
        )
        spk_trig_V = np.copy(spk_trig_I)

        # Create a support matrix of timepoints.
        t_vec = np.arange(-args.before, args.after, tr.dt, dtype = np.float32)
        assert len(t_vec) == spk_trig_I.shape[1], 'Support does not have same number of timesteps as data vectors.'
        spk_trig_t = np.tile(t_vec, (len(tr.spks), 1))
        del t_vec

        # Extract spikes.
        for j, spk in enumerate(tr.spks):

            start_grab_ind = max(0, (spk - int_DT_beforespk))
            stop_grab_ind = min(len(tr.I), (spk + int_DT_afterspk))

            start_ins_ind = start_grab_ind - (spk - int_DT_beforespk) # Should usually be zero.
            stop_ins_ind = (spk + int_DT_afterspk) - stop_grab_ind # Should usually be zero.
            if stop_ins_ind == 0:
                stop_ins_ind = spk_trig_I.shape[1]

            spk_trig_I[j, start_ins_ind:stop_ins_ind] = tr.I[start_grab_ind:stop_grab_ind]
            spk_trig_V[j, start_ins_ind:stop_ins_ind] = tr.V[start_grab_ind:stop_grab_ind]

        # Compute spike times and ISIs.
        spk_times = np.array(tr.spks) * tr.dt
        ISIs = np.concatenate([[np.nan], np.diff(spk_times)])

        # Save to main output dict.
        spike_triggered_slices['I'].extend([spk_trig_I[j, :] for j in range(spk_trig_I.shape[0])]) # Convert to lists while preserving single precision.
        spike_triggered_slices['V'].extend([spk_trig_V[j, :] for j in range(spk_trig_V.shape[0])])
        spike_triggered_slices['t'].extend([spk_trig_t[j, :] for j in range(spk_trig_t.shape[0])])

        spike_triggered_slices['t_spk'].extend(spk_times.tolist())
        spike_triggered_slices['t_since_spk'].extend(ISIs.tolist())
        spike_triggered_slices['names'].extend([expt.name] * len(tr.spks))


#%% RUN OUTPUT CHECKS AND SAVE

# Check that each list in output dict is same length.
ls_lens = []
for key, ls in spike_triggered_slices.iteritems():
    ls_lens.append(len(ls))
assert np.allclose(ls_lens, ls_lens[0]), 'Lists in output dict must all have equal length.'
del ls_lens

# Save output.
if args.verbose:
    print 'Saving output to {}'.format(args.output)
with open(args.output, 'wb') as f:
    pickle.dump(spike_triggered_slices, f)
    f.close()
if args.verbose:
    print 'Done!'

