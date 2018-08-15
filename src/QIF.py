#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


#%% DEFINE QIF CLASS

class QIF(object):

    def __init__(self,
        A = 0.04, B = 5, C = 140, V_peak = 0,
        a = 0.005, b = 0.2, c = -57., d = 2.,
        sigma = 2):

        self.A = A
        self.B = B
        self.C = C
        self.V_peak = V_peak
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sigma = sigma


    def simulate(self, I, V0, dt = 0.1):

        V = np.empty_like(I)
        U = np.empty_like(I)
        spks = np.zeros_like(I)

        scaled_rands_rtdt = self.sigma * np.random.normal(0, 1, size = I.shape) * np.sqrt(dt)

        V[0, :] = V0
        U[0, :] = 0

        for t in range(1, I.shape[0]):

            V_tm1 = V[t-1, :]
            U_tm1 = U[t-1, :]

            # Integrate voltage
            dV_deterministic = self.A * V_tm1**2 + self.B * V_tm1 + self.C - U_tm1 + I[t-1, :]
            V[t] = V_tm1 + dV_deterministic * dt + scaled_rands_rtdt[t-1, :]

            # Integrate recovery variable
            dU = self.a * (self.b * V_tm1 - U_tm1)
            U[t] = U_tm1 + dU * dt

            # Spiking rule
            spks_mask = V_tm1 >= self.V_peak
            spks[t-1, spks_mask] = 1

            V[t, spks_mask] = self.c
            U[t, spks_mask] = U_tm1[spks_mask] + self.d

        return V, U, spks


class QIFsim(object):

    def __init__(self, mod, I, V0, dt = 0.1):

        self.mod = mod
        self.dt = dt
        self.I = I

        self.V, self.U, self.spks = mod.simulate(I, V0, dt)

    @property
    def no_neurons(self):
        return self.I.shape[1]

    @property
    def t_vec(self):
        return np.arange(0, (self.I.shape[0] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self):
        return np.tile(self.t_vec[:, np.newaxis], (1, self.no_neurons))

    def plot(self, first_tr_only = True):

        spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.8])

        plt.figure()

        V_ax = plt.subplot(spec[1, :])
        if first_tr_only:
            plt.plot(self.t_vec, self.V[:, 0], 'k-', lw = 0.5)
        else:
            plt.plot(self.t_mat, self.V, 'k-', lw = 0.5)

        I_ax = plt.subplot(spec[0, :], sharex = V_ax)
        if first_tr_only:
            plt.plot(self.t_vec, self.I[:, 0], color = 'gray', lw = 0.5)
        else:
            plt.plot(self.t_mat, self.I, color = 'gray', lw = 0.5)
        plt.xticks([])

        raster_Ax = plt.subplot(spec[2, :], sharex = V_ax)
        for i in range(self.no_neurons):
            x = np.where(self.spks[:, i])[0] * self.dt
            plt.plot(x, [i for j in x], 'k|')

        plt.axis('off')

        plt.show()
