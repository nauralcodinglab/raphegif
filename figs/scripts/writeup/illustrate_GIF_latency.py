import os
os.chdir(os.path.join('..', '..', '..'))
print(os.getcwd())

import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
import ezephys.stimtools as st

from grr.Tools import timeToIndex, reprint


# LOAD MODELS

if __name__ == '__main__':
    MOD_PATH = os.path.join('data', 'models', '5HT')
    models = {}
    with open(os.path.join(MOD_PATH, '5HT_GIFs.lmod'), 'rb') as f:
        models['GIF'] = pickle.load(f)
        f.close()
    with open(os.path.join(MOD_PATH, '5HT_AugmentedGIFs.lmod'), 'rb') as f:
        models['AugmentedGIF'] = pickle.load(f)
        f.close()


# GENERATE STIMULI
step_params = {
    'dt': 0.1,
    'pre_duration': 2000.,  # ms
    'post_duration': 1000.  # ms
}
step_params['pre_amplitudes'] = [-0.02, -0.05, 0.1, -0.02, -0.05, 0.1, -0.02, -0.05, 0.1]
step_params['post_amplitudes'] = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03]

stimulus = []
for pre_amplitude, post_amplitude in zip(step_params['pre_amplitudes'], step_params['post_amplitudes']):
    stimulus.append(
        np.concatenate(
            (
                pre_amplitude * np.ones(timeToIndex(step_params['pre_duration'], step_params['dt'])),
                post_amplitude * np.ones(timeToIndex(step_params['post_duration'], step_params['dt']))
            ),
            axis=-1
        )
    )
stimulus = st.ArrayStimulus(stimulus, step_params['dt'])

latency_params = {
    'nb_reps': 500
}

def parfunc(mod):
    sample_traces_partial = {
        'model': [],
        'cell': [],
        'pre_amplitude': [],
        'post_amplitude': [],
        'trace': [],
    }
    spike_latencies_partial = {
        'model': [],
        'cell': [],
        'pre_amplitude': [],
        'post_amplitude': [],
        'latency': [],
        'premature_spike': []
    }

    for stim in range(stimulus.command.shape[0]):
        for i in range(latency_params['nb_reps']):
            reprint('Running {} {} simulations {:.1f}%'.format(type(mod), mod.name, 100. * i / latency_params['nb_reps']))
            t, V, eta_sum, V_T, spks = mod.simulate(stimulus.command[stim, :], mod.El)

            # Record sample trace for only one rep.
            if i == 0:
                sample_traces_partial['model'].append(type(mod))
                sample_traces_partial['cell'].append(mod.name)
                sample_traces_partial['pre_amplitude'].append(stimulus.command[stim, 0])
                sample_traces_partial['post_amplitude'].append(stimulus.command[stim, -1])
                sample_traces_partial['trace'].append({'t': t, 'V': V, 'spks': spks})

            # Record latency parameters.
            spike_latencies_partial['model'].append(type(mod))
            spike_latencies_partial['cell'].append(mod.name)
            spike_latencies_partial['pre_amplitude'].append(stimulus.command[stim, 0])
            spike_latencies_partial['post_amplitude'].append(stimulus.command[stim, -1])

            # Record latency results.
            if not any(np.asarray(spks) < step_params['pre_duration']):
                # No spikes before pulse start.
                spike_latencies_partial['premature_spike'].append(False)
                eligible_spks = np.asarray(spks)[np.asarray(spks) > step_params['pre_duration']]
                if len(eligible_spks) > 0:
                    # There is a spike after pulse start.
                    spike_latencies_partial['latency'].append(np.min(eligible_spks))
                else:
                    # No spikes after pulse start.
                    spike_latencies_partial['latency'].append(np.nan)
            else:
                # Spike before pulse start.
                spike_latencies_partial['latency'].append(np.nan)
                spike_latencies_partial['premature_spike'].append(True)

    return {'spike_latencies': spike_latencies_partial, 'sample_traces': sample_traces_partial}


if __name__ == '__main__':

    # RUN SIMULATIONS

    sample_traces = []
    spike_latencies = []

    pool = mp.Pool(mp.cpu_count())

    try:
        for modtype in models:
            tmp_result = pool.map(parfunc, models[modtype], 1)
            sample_traces.extend([res['sample_traces'] for res in tmp_result])
            spike_latencies.extend([res['spike_latencies'] for res in tmp_result])
    finally:
        pool.close()
        pool.join()

    sample_traces = pd.concat([pd.DataFrame(x) for x in sample_traces], ignore_index=True)
    spike_latencies = pd.concat([pd.DataFrame(x) for x in spike_latencies], ignore_index=True)

    # SAVE RESULTS

    SAVE_PATH = os.path.join('data', 'simulations', 'GIF_latency')
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    with open(os.path.join(SAVE_PATH, 'sample_traces.pkl'), 'wb') as f:
        pickle.dump(sample_traces, f)
        f.close()

    spike_latencies.to_csv(os.path.join(SAVE_PATH, 'latency.csv'), index=False)

