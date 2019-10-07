#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from grr.cell_class import Cell, Recording


#%% DEFINE SPK TIME PREDICTOR

class OhmicSpkPredictor(object):

    def __init__(self):
        pass

    def add_recordings(self, recs, baseline = None, ss = None, V_channel = 0, I_channel = 1, dt = 0.1):

        """
        Recs should be a list of Recording objects with dimensionality [channel, time, sweep]
        """

        recs_arr = np.array(recs)
        self.V = deepcopy(recs_arr[:, V_channel, :, :])
        self.I = deepcopy(recs_arr[:, I_channel, :, :])

        self.dt = dt

        if type(recs[0]) is Recording and (baseline is not None and ss is not None):

            self.Rins = []

            for rec in recs:
                tmp = rec.fit_test_pulse(baseline, ss, V_clamp = False, verbose = False)
                self.Rins.append(tmp['R_input'].mean())

    @property
    def t_vec(self):

        return np.arange(0, (self.V.shape[1] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self, reps = 'default'):

        if reps == 'default':
            reps = self.V.shape[2]

        return np.tile(self.t_vec[:, np.newaxis], (1, reps))


    def phase_plot(self):

        dV = np.gradient(self.V, axis = 1)

        plt.figure()
        for i in range(self.V.shape[0]):
            plt.plot(
                self.V[i, :, :], dV[i, :, :] / self.dt,
                color = cm.coolwarm(i / self.V.shape[0]),
                alpha = 1/self.V.shape[0]
            )

        plt.xlabel('V (mV)')
        plt.ylabel('dV/dt (mV/ms)')

        plt.show()


    def scrape_data(self, V0_range = (2495, 2595), Vinf_range = (2750, 2800), baseline_range = (0, 100),
        quiescent_until = 2600, dVdt_thresh = 10):

        """
        Scrape spks, V0, and Vinf from a set of recordings.

        Inputs:

            V0_range: tuple of two floats
            --  Time (ms) range from which to get V0

            Vinf_range: tuple of two floats
            --  Time (ms) range from which to calculate Vinf *based on I_probe*

            quiescent_until: float
            --  Discard sweeps that have spikes before this point (ms).

            dVdt_thresh: float
            --  Voltage derivative threshold to use for spk detection (mV/ms).

        Places all spk times together in a big honking list. Complementary big honking lists of V0 and Vinf are generated.
        Only takes the first spk after `quiescent_until` from each sweep.
        """

        V0_slice        = slice(int(V0_range[0] / self.dt), int(V0_range[1] / self.dt))
        Vinf_slice      = slice(int(Vinf_range[0] / self.dt), int(Vinf_range[1] / self.dt))
        baseline_slice  = slice(int(baseline_range[0] / self.dt), int(baseline_range[1] / self.dt))

        dVdt = np.gradient(self.V, axis = 1) / self.dt

        spks = []
        V0 = []
        Vinf = []

        for rec_ind in range(dVdt.shape[0]):

            V_baseline = self.V[:, baseline_slice, :].mean()
            I_baseline = self.I[:, baseline_slice, :].mean()

            for sw_ind in range(dVdt.shape[2]):

                # Get spk inds for this recording.
                spks_i = np.where(dVdt[rec_ind, :, sw_ind] > dVdt_thresh)[0] * self.dt

                # Skip sweep if premature spks are detected.
                if any(spks_i < quiescent_until):
                    continue

                # Skip sweep if no spks found.
                if len(spks_i) < 1:
                    continue

                # Extract V0
                V0_i = self.V[rec_ind, V0_slice, sw_ind].mean()

                # Extract Vinf
                I_probe = self.I[rec_ind, Vinf_slice, sw_ind].mean()
                Vinf_i = self.Rins[rec_ind] * (I_probe - I_baseline) * 1e-3 + V_baseline

                # Assign output
                spks.append(spks_i[0] - quiescent_until)
                V0.append(V0_i)
                Vinf.append(Vinf_i)

        # Store in class attributes
        self.spks   = np.array(spks)
        self.V0     = np.array(V0)
        self.Vinf   = np.array(Vinf)


    def plot(self):

        plt.figure()

        spec = plt.GridSpec(2, 1, height_ratios = [2, 1])

        plt.subplot(spec[0, :])
        for i in range(self.V.shape[0]):
            plt.plot(
                self.t_mat,
                self.V[i, :, :],
                color = cm.coolwarm(i / self.V.shape[0]),
                alpha = 1/self.V.shape[0]
            )
        plt.plot(self.spks, [-30 for i in self.spks], 'bx')

        plt.subplot(spec[1, :])
        for i in range(self.I.shape[0]):
            plt.plot(
                self.t_mat,
                self.I[i, :, :],
                color = cm.coolwarm(i / self.V.shape[0]),
                alpha = 1/self.V.shape[0]
            )


    def fit_spks(self, thresh_guesses = 'default', Vinf_guesses = 'default', force_tau = None, verbose = False):

        if thresh_guesses == 'default':
            thresh_guesses = np.linspace(-50, 0, 250)

        if Vinf_guesses == 'default':
            margin = self.Vinf.mean() * 2.
            Vinf_guesses = np.linspace(
                self.Vinf.mean() - margin,
                self.Vinf.mean() + margin,
                250
            )

        tau_est = []
        Vinf_ = []
        thresh_ = []
        SSE = []

        y = np.array(self.spks)

        for i, thresh in enumerate(thresh_guesses):
            for j, Vinf in enumerate(Vinf_guesses):

                if verbose:
                    print '\rFitting {:.1f}%'.format(100 * (i+1)/len(thresh_guesses)),

                X = - np.log( (thresh - Vinf) / (self.V0 - Vinf) )
                if force_tau is None:
                    XTX = np.dot(X.T, X)
                    XTX_inv = 1/XTX
                    XTY = np.dot(X.T, y)

                    b = np.dot(XTX_inv, XTY)
                else:
                    b = force_tau

                tau_est.append(b)

                yhat = np.dot(X, b)
                SSE.append(np.sum( (y - yhat)**2 ))

                Vinf_.append(Vinf)
                thresh_.append(thresh)

        self.tau = tau_est[np.nanargmin(SSE)]
        self.thresh = thresh_[np.nanargmin(SSE)]
        self.Vinf_est = Vinf_[np.nanargmin(SSE)]

        return tau_est, SSE


    def predict_spks(self, **kwargs):

        """
        Valid kwargs are: tau, thresh, V0, Vinf
        """

        valid_kwargs = ['tau', 'thresh', 'V0', 'Vinf']
        if any([key not in valid_kwargs for key in kwargs.keys()]):
            raise NameError('Valid kwargs are: {}'.format(', '.join(valid_kwargs)))

        tau         = kwargs.get('tau', self.tau)
        thresh      = kwargs.get('thresh', self.thresh)
        V0          = kwargs.get('V0', self.V0)
        Vinf        = kwargs.get('Vinf', self.Vinf)

        spk_prediction = - tau * np.log( (thresh - Vinf) / (V0 - Vinf) )

        return spk_prediction
