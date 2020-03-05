import pickle
import argparse

import numpy as np
import ezephys.stimtools as st


# PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument(
    'output',
    help='Path to save output. '
    'Pickled dict with ser_input and gaba_input as attributes.',
)

parser.add_argument('--dt', type=float, default=0.1, help='Timestep of input.')
parser.add_argument(
    '-v',
    '--verbose',
    help='Print more information about progress.',
    action='store_true',
)

args = parser.parse_args()


# ASSEMBLE STIMULUS

synaptic_current_params = {
    'amplitude': 0.015,
    'rise_tau': 2,
    'decay_tau': 50
}
num_synapses = 50
network_size = {
    'ser': 600,
    'som': 400
}
jitters = [500, 100, 50, 10]
padding = st.StepStimulus([250.], [0.], dt=args.dt)

def generate_synaptic_burst_stimulus():
    # Can be called repeatedly to generate random bursts of input.
    # Returns a vector.

    synaptic_bursts = [padding]
    for jitter, post_sigma in zip(jitters, [4, 20, 40, 80]):
        synaptic_trigger = st.DiscreteJitteredEvent(jitter, dt=args.dt)

        synaptic_bursts.append(
            st.ConvolvedStimulus(
                0,
                st.BiexponentialSynapticKernel(
                    synaptic_current_params['amplitude'],
                    synaptic_current_params['rise_tau'],
                    synaptic_current_params['decay_tau'],
                    size_method='amplitude',
                    dt=args.dt
                ),
                synaptic_trigger.sample(50, (2, post_sigma)).sum(axis=0),  # This is stochastic.
            )
        )

    synaptic_bursts.append(padding)
    synaptic_bursts = st.concatenate(synaptic_bursts)

    # Assert that command array is either a column or row vector and therefore
    # can be flattened.
    if np.ndim(synaptic_bursts.command) > 1:
        assert np.ndim(synaptic_bursts.command) == 2
        assert synaptic_bursts.command.size == np.max(np.shape(synaptic_bursts.command))

    return synaptic_bursts.command.flatten()


# GENERATE STOCHASTIC BURSTS
# Each neuron gets unique input.

np.random.seed(42)

synaptic_burst_stimulus = {
    'ser_input': np.array([generate_synaptic_burst_stimulus() for i in range(network_size['ser'])])[np.newaxis, :, :],
    'gaba_input': np.array([generate_synaptic_burst_stimulus() for i in range(network_size['som'])])[np.newaxis, :, :],
    'metaparams': {
        'jitters': jitters,
        'synaptic_current': synaptic_current_params,
        'num_synapses': num_synapses
    }
}


# SAVE TO FILE
if args.verbose:
    print('Saving synaptic burst stimulus to {}'.format(args.output))
with open(args.output, 'wb') as f:
    pickle.dump(synaptic_burst_stimulus, f)
    f.close()
if args.verbose:
    print('Done!')
