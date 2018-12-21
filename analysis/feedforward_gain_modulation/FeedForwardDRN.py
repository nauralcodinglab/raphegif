#%% IMPORT MODULES

from __future__ import division

import copy

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./analysis/spk_timing/IA_mod/')
import IAmod


#%% DEFINE HANDY FUNCTIONS

class SynapticKernel(object):

    def __init__(self, kernel_type = None, dt = 0.001, **params):

        self._defined_kernel_types = ['biexponential', 'biexp', 'alpha']
        self.dt = dt

        if kernel_type is not None:

            if kernel_type in ['biexponential', 'biexp']:
                self.generate_biexp(**params)
            elif kernel_type in ['alpha']:
                self.generate_alpha(**params)
            else:
                raise ValueError(
                    'Got undefined kernel_type {}.\n'
                    'Expected one of: {}.'.format(
                        kernel_type, ', '.join(self._defined_kernel_types)
                        )
                    )
        else:
            self.kernel_type = kernel_type

    def __repr__(self):
        return '<{} with {} type kernel at {}>'.format(type(self).__name__, self.kernel_type, hex(id(self)))


    ### Methods for support structures.
    def generate_t_vec(self, kernel_len):
        return np.arange(0, kernel_len, self.dt)

    @property
    def t_vec(self):
        return np.arange(0, (len(self.kernel) - 0.5) * self.dt, self.dt)

    ### Methods to generate kernels.
    def generate_biexp(self, tau_rise, tau_decay, ampli, kernel_len):
        t_supp = self.generate_t_vec(kernel_len)
        kernel = np.exp(-t_supp/tau_rise) - np.exp(-t_supp/tau_decay)
        kernel = ampli * kernel / kernel.min()

        self.kernel_type = 'biexp'
        self.kernel = kernel
        self.kernel_params = {
            'tau_rise': tau_rise,
            'tau_decay': tau_decay,
            'ampli': ampli
        }

        return kernel

    def generate_alpha(self, tau, ampli, kernel_len):
        t_supp = self.generate_t_vec(kernel_len)
        kernel = t_supp / tau * np.exp(-t_supp / tau)
        kernel = ampli * kernel / kernel.max()

        self.kernel_type = 'alpha'
        self.kernel = kernel
        self.kernel_params = {
            'tau': tau,
            'ampli': ampli
        }

        return kernel

    ### Misc.
    @property
    def centered_kernel(self):
        return np.concatenate([np.zeros_like(self.kernel), self.kernel])

    @property
    def centered_t_vec(self):
        return np.concatenate([self.t_vec, self.t_vec + self.t_vec[-1]])

    def pad(self, pad_length):
        return np.concatenate([np.zeros(int(pad_length / self.dt)), self.kernel])

    def tile(self, reps, pad_length = None, centered = False):
        if (not centered) and pad_length is None:
            tmp_kernel = self.kernel
        elif (not centered) and pad_length is not None:
            tmp_kernel = self.pad(pad_length)
        elif centered and pad_length is None:
            tmp_kernel = self.centered_kernel
        else:
            raise ValueError(
                'Invalid input combination.'
                'Either `pad_length` or `centered` can be set, but not both.'
            )

        return np.tile(tmp_kernel[:, np.newaxis], (1, reps))

    ### Methods for inspection.
    def plot(self, pad_length = None, centered = False):

        if (not centered) and pad_length is None:
            title_prefix = ''
            tmp_kernel = self.kernel
            tmp_supp = self.t_vec
        elif (not centered) and pad_length is not None:
            raise NotImplementedError('Plotting method not implemented for padded kernels.')
        elif centered and pad_length is None:
            title_prefix = 'Centered'
            tmp_kernel = self.centered_kernel
            tmp_supp = self.centered_t_vec
        else:
            raise ValueError(
                'Invalid input combination.'
                'Either `pad_length` or `centered` can be set, but not both.'
            )

        plt.figure()

        plt.subplot(111)
        plt.title('{} {} kernel'.format(title_prefix, self.kernel_type))
        plt.plot(tmp_supp, tmp_kernel, 'k-')
        plt.ylabel('Amplitude')
        plt.xlabel('Time')

        plt.show()


#%% DEFINE MAIN FEEDFORWARDDRN NETWORK MODEL CLASS

class FeedForwardDRN(object):

    """
    DRN network model with feed-forward inhibition from GABA cells onto 5HT (ser)
    cells.
    """

    def __init__(self, name = 'unnamed', dt = 0.001):

        self.name = name
        self.dt = dt

        self.propagation_delay = 0

    ### Construct model components.
    def construct_ser_mod(self, ga, tau_h, sigma_noise, tau_eff):
        self.ser_mod = IAmod.IAmod(ga, tau_h, sigma_noise, tau_eff)

    def construct_gaba_mod(self, ga, tau_h, sigma_noise, tau_eff):
        self.gaba_mod = IAmod.IAmod(ga, tau_h, sigma_noise, tau_eff)

    def construct_gaba_kernel(self, tau_rise, tau_decay, ampli, kernel_len):
        gaba_kernel_obj = SynapticKernel(
            'biexp', dt = self.dt,
            tau_rise = tau_rise, tau_decay = tau_decay,
            ampli = ampli, kernel_len = kernel_len
        )

        self._gaba_kernel_obj = gaba_kernel_obj

    @property
    def gaba_kernel(self):
        return self._gaba_kernel_obj.kernel

    ### Attach input.
    def set_ser_Vin(self, Vin):
        self.ser_Vin = Vin

    @property
    def ser_no_neurons(self):
        return self.ser_Vin.shape[1]

    def set_gaba_Vin(self, Vin):
        self.gaba_Vin = Vin

    @property
    def gaba_no_neurons(self):
        return self.gaba_Vin.shape[1]

    ### Set other parameters.
    def set_propagation_delay(self, propagation_delay):
        self.propagation_delay = propagation_delay

    ### Methods for simulation.
    def _simulate_gaba(self, Vin, V0 = -60, random_seed = 42):
        """
        Return IAmod.Simulation using instance gaba_mod and dt.
        """
        return IAmod.Simulation(self.gaba_mod, V0, Vin, self.dt, random_seed)

    def _convolve_gaba_spks_all_to_all(self, propagation_delay = 0, normalize_neurons = True):
        """
        Return an all-to-all feed-forward GABA -> 5HT signal.

        Take the sum of spks across GABA population for each time bin,
        convolve the result with the IPSC kernel, and tile the result
        to the number of 5HT neurons. Returns the result.

        Inputs:
            propagation_delay (float)
                --  Delay between GABA spk and onset of IPSC
            normalize_neurons (bool)
                --  Normalize the amplitude of the IPSC kernel to the no. of GABA neurons
        """

        delay_ind = int(propagation_delay / self.dt)

        gaba_spks = self.gaba_sim.spks.sum(axis = 1)
        gaba_out = np.zeros_like(gaba_spks, dtype = np.float64)

        for i, spk_cnt in enumerate(gaba_spks):
            t_ind_tmp = delay_ind + i
            if spk_cnt == 0 :
                continue
            elif t_ind_tmp >= len(gaba_spks):
                break
            else:
                out_end_ind = min(len(gaba_out), t_ind_tmp + len(self.gaba_kernel))
                kernel_end_ind = min(out_end_ind - t_ind_tmp, len(self.gaba_kernel))
                gaba_out[t_ind_tmp:out_end_ind] += spk_cnt * self.gaba_kernel[:kernel_end_ind] / self.gaba_no_neurons

        gaba_out = np.tile(gaba_out[:, np.newaxis], (1, self.ser_no_neurons))

        return gaba_out

    def _simulate_ser(self, Vin, V0 = -60, random_seed = 43):
        """
        Return IAmod.Simulation using instance ser_mod and dt.
        """
        return IAmod.Simulation(self.ser_mod, V0, Vin, self.dt, random_seed)

    def simulate(self, feed_forward = True, verbose = True):
        """
        Execute DRN simulation with or without GABAergic feed-forward inhibition.
        """

        if verbose:
            print('Starting {} simulation.\nSimulating GABA neurons...'.format(self.name))
        self.gaba_sim = self._simulate_gaba(self.gaba_Vin)

        if feed_forward:
            if verbose:
                print('Convolving GABA spks with IPSC kernel...')
            self.gaba_out = self._convolve_gaba_spks_all_to_all(self.propagation_delay)
            ser_net_Vin = self.ser_Vin + self.gaba_out
        else:
            self.gaba_out = None
            ser_net_Vin = self.ser_Vin

        if verbose:
            print('Simulating ser neurons...')
        self.ser_sim = self._simulate_ser(ser_net_Vin)
        if verbose:
            print('Done {}!\n'.format(self.name))

    ### Misc.
    def copy(self, rename = None):
        """
        Return a deep copy of the current instance.
        """
        tmp = copy.deepcopy(self)
        if rename is not None:
            tmp.name = rename
        return tmp

    ### Methods for visualization and analysis.
    def gaba_psth(self, window_width):
        return self.gaba_sim.PSTH(window_width)

    def ser_psth(self, window_width):
        return self.ser_sim.PSTH(window_width)

    def plot_psth(self, window_width, return_fig = False):

        plt.figure()

        plt.suptitle(self.name)

        plt.subplot(111)
        plt.plot(self.gaba_sim.t_vec, self.gaba_sim.PSTH(window_width), 'g-', label = 'GABA')
        plt.plot(self.ser_sim.t_vec, self.ser_sim.PSTH(window_width), 'b-', label = '5HT')
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)

        if return_fig: return plt.gcf()



    def plot_rasters(self, return_fig = False):

        plt.figure()

        plt.suptitle(self.name)

        gaba_raster = plt.subplot(211)
        plt.title('GABA')
        where_spks = np.where(self.gaba_sim.spks)
        plt.plot(where_spks[0] * self.dt, where_spks[1], 'g|', markersize = 0.5)
        plt.ylabel('Neuron no.')

        ser_raster = plt.subplot(212, sharex = gaba_raster)
        plt.title('5HT')
        where_spks = np.where(self.ser_sim.spks)
        plt.plot(where_spks[0] * self.dt, where_spks[1], 'r|', markersize = 0.5)
        plt.ylabel('Neuron no.')
        plt.xlabel('Time (tau)')

        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)

        if return_fig: return plt.gcf()

    def plot_traces(self, no_neurons = 20, return_fig = False):

        plt.figure()

        plt.suptitle(self.name)

        gaba_traces = plt.subplot(211)
        plt.title('GABA')
        plt.plot(
            self.gaba_sim.t_mat[:, :no_neurons], self.gaba_sim.V[:, :no_neurons],
            'g-', lw = 0.5, alpha = min(1, 5/no_neurons)
        )
        plt.ylabel('V (mV)')

        ser_traces = plt.subplot(212, sharex = gaba_traces)
        plt.title('5HT')
        plt.plot(
            self.ser_sim.t_mat[:, :no_neurons], self.ser_sim.V[:, :no_neurons],
            'r-', lw = 0.5, alpha = min(1, 5/no_neurons)
        )
        plt.ylabel('V (mV)')
        plt.xlabel('Time (tau)')

        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)

        if return_fig: return plt.gcf()

    def plot_gaba_kernel(self):
        self._gaba_kernel_obj.plot()
