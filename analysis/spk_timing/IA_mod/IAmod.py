#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


#%% DEFINE MODEL CLASS

class IAmod(object):

    def __init__(self, ga, tau_h, sigma_noise, tau_eff = 1):

        self.tau_eff = tau_eff

        self.ga = ga

        self.El = -70.
        self.Ea = -101.

        self.m_A = 1.61
        self.m_Vhalf = -23.7
        self.m_k = 0.0985

        self.h_Vhalf = -59.2
        self.h_k = -0.165
        self.h_tau = tau_h
        self.h_A = 1.03

        self.sigma_noise = sigma_noise

        self.theta = -20.
        self.vreset = -55.

    def m_inf(self, V):
        return self.m_A / (1 + np.exp( -self.m_k * (V - self.m_Vhalf) ))

    def h_inf(self, V):
        return self.h_A / (1 + np.exp( -self.h_k * (V - self.h_Vhalf) ))


    def simulate(self, V0, Vin, dt = 1e-3, random_seed = 42):

        # Define functions
        @nb.vectorize(['f8(f8)'], cache = True)
        def m_inf(V):
            return 1 / (1 + np.exp( -self.m_k * (V - self.m_Vhalf) ))

        @nb.vectorize(['f8(f8)'], cache = True)
        def h_inf(V):
            return 1 / (1 + np.exp( -self.h_k * (V - self.h_Vhalf) ))

        # Locally define variables.
        tau_eff     = deepcopy(self.tau_eff)
        tau_h       = deepcopy(self.h_tau)
        ga          = deepcopy(self.ga)
        El          = deepcopy(self.El)
        Ea          = deepcopy(self.Ea)
        theta       = deepcopy(self.theta)
        vreset      = deepcopy(self.vreset)

        # Allocate arrays for output
        V_mat = np.empty(Vin.shape, dtype = np.float64)
        m_mat = np.empty_like(V_mat)
        h_mat = np.empty_like(V_mat)
        spks_mat = np.zeros_like(V_mat, dtype = np.bool)

        # Set initial conditions.
        V_mat[0, :] = V0
        m_mat[0, :] = m_inf(V0)
        h_mat[0, :] = h_inf(V0)

        # Generate random values for noise in the dynamics.
        np.random.seed(random_seed)
        V_noise = self.sigma_noise * np.random.normal(size = Vin.shape)

        for t in range(1, len(V_mat)):

            # Integrate gates.
            m_mat[t] = m_inf(V_mat[t-1, :])

            dh = (h_inf(V_mat[t-1, :]) - h_mat[t-1, :]) / tau_h
            h_mat[t] = h_mat[t-1, :] + dh * dt

            # Integrate voltage.
            dV = -(V_mat[t-1, :] - El) - ga * m_mat[t-1, :] * h_mat[t-1, :] * (V_mat[t-1, :] - Ea) + Vin[t-1, :]
            V_mat[t] = V_mat[t-1, :] + (dV * dt + V_noise[t-1, :] * np.sqrt(dt)) / tau_eff

            # Detect spiking neurons, log spikes, and reset voltage where applicable.
            spiking_neurons = V_mat[t-1, :] >= theta
            spks_mat[t-1, spiking_neurons] = True
            V_mat[t, spiking_neurons] = vreset

        return V_mat, spks_mat, m_mat, h_mat

    def ss_clamp(self, V):

        """
        Return the amount of input voltage required to keep the cell at a specified target voltage at steady-state.
        """

        return (V - self.El) + self.ga * self.m_inf(V) * self.h_inf(V) * (V - self.Ea)


class Simulation(object):

    def __init__(self, mod, V0, Vin, dt = 1e-3, random_seed = 42):

        V_mat, spks_mat, m_mat, h_mat = mod.simulate(V0, Vin, dt, random_seed)

        self.V      = V_mat
        self.spks   = spks_mat
        self.m      = m_mat
        self.h      = h_mat

        self.dt     = dt


    @property
    def no_neurons(self):
        return self.V.shape[1]

    @property
    def t_vec(self):
        return np.arange(0, (self.V.shape[0] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self):
        return np.tile(self.t_vec[:, np.newaxis], (1, self.no_neurons))


    def get_spk_latencies(self):

        first_spk = np.zeros(self.no_neurons)

        for i in range(self.no_neurons):

            spk_locs = np.where(self.spks[:, i])[0]

            if len(spk_locs) > 0:
                first_spk[i] = spk_locs[0] * self.dt
            else:
                first_spk[i] = np.NaN

        return first_spk

    def PSTH(self, window_width):
        """
        Obtain the population firing rate with a resolution of `window_width`.
        """
        kernel = np.ones(int(window_width / self.dt)) / (window_width * self.no_neurons)
        spks_sum = self.spks.sum(axis = 1)
        psth = np.convolve(spks_sum, kernel, 'same')
        return psth


    def simple_plot(self):

        alpha = min([5/self.no_neurons, 1])

        fig = plt.figure()

        v_ax = fig.add_subplot(312)
        v_ax.set_title('Subthreshold voltage')
        v_ax.plot(self.t_mat, self.V, 'k-', alpha = alpha)
        v_ax.set_ylabel('$V$ (mV)')

        raster_ax = fig.add_subplot(311, sharex = v_ax)
        raster_ax.set_title('Spike raster')
        where_spks = np.where(self.spks)
        raster_ax.plot(where_spks[0] * self.dt, where_spks[1], 'k|', markersize = 0.5)
        raster_ax.set_ylabel('Repetition number')

        gating_ax = fig.add_subplot(313, sharex = v_ax)
        gating_ax.set_title('$I_A$ gating variables')
        gating_ax.plot(self.t_vec, self.m[:, 0], '-', color = (0.8, 0.2, 0.2), alpha = alpha, label = 'm')
        gating_ax.plot(self.t_mat[:, 1:], self.m[:, 1:], '-', color = (0.8, 0.2, 0.2), alpha = alpha)
        gating_ax.plot(self.t_vec, self.h[:, 0], '-', color = (0.2, 0.8, 0.2), alpha = alpha, label = 'h')
        gating_ax.plot(self.t_mat[:, 1:], self.h[:, 1:], '-', color = (0.2, 0.8, 0.2), alpha = alpha)
        gating_ax.legend()
        gating_ax.set_ylabel('$g/g_{{max}}$')

        fig.tight_layout()
