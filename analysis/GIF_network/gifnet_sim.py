"""Script for running parallelized GIFnet simulations.

INTENDED TO BE RUN FROM COMMAND LINE
"""

#%% IMPORT MODULES

from __future__ import division

import argparse
import pickle
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pandas as pd

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
from FeedForwardDRN import SynapticKernel
import src.GIF_network as gfn
from src.Tools import generateOUprocess, timeToIntVec

#%% PARSE COMMANDLINE ARGUMENTS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        help = 'Path to GIFnet model used to run the simulation.'
    )
    parser.add_argument(
        'destination_path',
        help = 'Filepath for storing the results of the simulation.'
    )
    parser.add_argument(
        '--input_tau', type = float,
        help = 'Time constant of synaptic input alpha kernel.',
        default = 15.
    )
    parser.add_argument(
        '--processes', type = int,
        help = 'Number of processes to spawn to accelerate simulations.',
        default = 4.
    )
    parser.add_argument(
        '-v', '--verbose',
        help = 'Increase output verbosity.',
        action = 'store_true'
    )

    args = parser.parse_args()


#%% LOAD MODEL AND INITIALIZE VARS NEEDED TO RUN SIMULATION

if __name__ == '__main__':

    ### Load model.
    with open(args.model, 'rb') as f:
        gifnet_mod = pickle.load(f)
        f.close()

    ### Define distal input.
    distal_in = SynapticKernel(
        'alpha', tau = args.input_tau, ampli = 1, kernel_len = 500, dt = gifnet_mod.dt
    ).centered_kernel

    scales = np.linspace(0.010, 0.200, 15)

    ### Initialize random components
    OU_noise = {
        'ser': np.array(
            [generateOUprocess(len(distal_in) * gifnet_mod.dt, 3, 0, 0.020, gifnet_mod.dt, random_seed = int(i + 1)) for i in range(gifnet_mod.connectivity_matrix.shape[0])]
        ),
        'gaba': np.array(
            [generateOUprocess(len(distal_in) * gifnet_mod.dt, 3, 0, 0.020, gifnet_mod.dt, random_seed = int((i + 1) / np.pi)) for i in range(gifnet_mod.connectivity_matrix.shape[1])]
        )
    }


#%% DEFINE A FUNCTION TO RANDOMIZE MODEL COEFFICIENTS

def fuzz_parameters(model):
    """Randomize GIF model parameters.
    """

    if not hasattr(model, 'param_sds'):
        raise AttributeError('model does not have required `param_sds` attribute.')

    model_cpy = deepcopy(model)

    for attr_ in model_cpy.param_sds.keys():
        vars(model_cpy)[attr_] = np.random.normal(getattr(model_cpy, attr_), model_cpy.param_sds[attr_])

    return model_cpy


#%% DEFINE A MAIN FUNC TO RUN THE SIMULATION IN PARALLEL

if __name__ == '__main__':

    def main(scale):
        """Main method for simulating GIFnet.
        Intended to be run in parallel using mp.Pool.map.
        """

        no_ser_neurons = gifnet_mod.connectivity_matrix.shape[0]
        no_gaba_neurons = gifnet_mod.connectivity_matrix.shape[1]

        I = scale * distal_in

        # Iterate over GABA neurons providing input to a single 5HT cell.
        gaba_outmat = np.empty((no_gaba_neurons, len(distal_in)), dtype = np.float32)
        for gaba_no in range(no_gaba_neurons):
            if hasattr(gifnet_mod.gaba_mod, 'param_sds'):
                tmp_gaba_mod = fuzz_parameters(gifnet_mod.gaba_mod)
            else:
                tmp_gaba_mod = gifnet_mod.gaba_mod

            t, V, eta, v_T, spks = tmp_gaba_mod.simulate(
                I + OU_noise['gaba'][gaba_no, :], tmp_gaba_mod.El
            )
            gaba_outmat[gaba_no, :] = np.convolve(
                timeToIntVec(spks, distal_in.shape[0] * gifnet_mod.dt, gifnet_mod.dt), gifnet_mod.gaba_kernel, 'same'
            ).astype(np.float32)

            # Save a sample trace.
            if gaba_no == 0:
                gaba_ex = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

            if args.verbose:
                print '\r{} simulating GABA neurons for scale {:.4f} - {:.1f}%'.format(
                    gifnet_mod.name, scale, 100 * (gaba_no + 1) / no_gaba_neurons
                ),
        if args.verbose: print '\n',

        # Add propagation delay to GABA input.
        gaba_outmat = np.roll(gaba_outmat, gifnet_mod.propagation_delay, axis = 1)
        gaba_outmat[:, :gifnet_mod.propagation_delay] = 0

        # Transform GABA output into 5HT input using connectivity matrix.
        gaba_inmat = np.dot(gifnet_mod.connectivity_matrix, gaba_outmat)

        # Allocate arrays for 5HT spiketrains.
        ser_spkvecs = {
            'ib': np.zeros_like(distal_in, dtype = np.int16),
            'reg': np.zeros_like(distal_in, dtype = np.int16)
        }
        # Create a dict to hold 5HT example traces.
        ser_examples = {}

        for ser_no in range(no_ser_neurons):
            I_tmp = I + OU_noise['ser'][ser_no, :]

            if hasattr(gifnet_mod.ser_mod, 'param_sds'):
                tmp_ser_mod = fuzz_parameters(gifnet_mod.ser_mod)
            else:
                tmp_ser_mod = gifnet_mod.ser_mod

            # Try with and without feed-forward inhibition
            for ib_status, ib_multiplier in zip(('ib', 'reg'), (1, 0)):
                t, V, eta, v_T, spks = tmp_ser_mod.simulate(
                    I_tmp + gaba_inmat[ser_no, :] * ib_multiplier,
                    tmp_ser_mod.El
                )
                ser_spkvecs[ib_status] += timeToIntVec(spks, distal_in.shape[0] * gifnet_mod.dt, gifnet_mod.dt)

                # Save a sample trace.
                if ser_no == 0:
                    ser_examples[ib_status] = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

            if args.verbose:
                print '\r{} simulating 5HT neurons for scale {:.4f} - {:.1f}%'.format(
                    gifnet_mod.name, scale, 100 * (ser_no + 1) / no_ser_neurons
                ),
        if args.verbose: print '\n'


        ff_results_tmp = pd.DataFrame(columns = ['Scale', 'Inhibition', 'Input', 'ser spks', 'gaba in', 'ser ex', 'gaba ex'])
        for ib_status in ('ib', 'reg'):
            row = {
                'Scale': scale,
                'Inhibition': ib_status,
                'Input': I,
                'ser spks': ser_spkvecs[ib_status],
                'gaba in': gaba_inmat,
                'ser ex': ser_examples[ib_status],
                'gaba ex': gaba_ex
            }
            ff_results_tmp = ff_results_tmp.append(row, ignore_index = True)

        return ff_results_tmp


#%% EXECUTE MAIN FUNC IN PARALLEL

if __name__ == '__main__':

    p = mp.Pool(int(args.processes))
    results_tmp = p.map(main, scales)
    p.close()

    if args.verbose:
        print 'Done simulation for {}! Saving output...'.format(gifnet_mod.name)

    results = pd.concat(results_tmp, axis = 0, ignore_index = True)


    with open(args.destination_path, 'wb') as f:
        pickle.dump(results, f)
        f.close()

    if args.verbose:
        print 'Finished {}!'.format(gifnet_mod.name)

