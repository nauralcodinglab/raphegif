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

    def add_recordings(self, recs, baseline = None, ss = None, tau = None, V_channel = 0, I_channel = 1, dt = 0.1):

        """
        Recs should be a list of Recording objects with dimensionality [channel, time, sweep]
        """

        self.tau = None
        self.thresh = None
        self.V0 = None
        self.Vinf = None

        recs_arr = np.array(recs)
        self.V = deepcopy(recs_arr[:, V_channel, :, :])
        self.I = deepcopy(recs_arr[:, I_channel, :, :])

        self.dt = dt

        if type(recs[0]) is Recording and (baseline is not None and ss is not None):

            self.Rins = []
            self.taus = []

            for rec in recs:
                if tau is None:
                    tmp = rec.fit_test_pulse(baseline, ss, V_clamp = False, verbose = False, V_chan=V_channel, I_chan=I_channel)
                    self.Rins.append(tmp['R_input'].mean())
                else:
                    tmp = rec.fit_test_pulse(baseline, ss, tau = tau, V_clamp = False, V_chan=V_channel, I_chan=I_channel, verbose = False, plot_tau=True)
                    self.Rins.append(tmp['R_input'].mean())
                    self.taus.append(tmp['tau'])

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
        quiescent_until = 2647, dVdt_thresh = 10, exclude_above = -55):

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

                if V0_i > exclude_above:
                    continue

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

                if force_tau is None:
                    X = - np.log( (thresh - Vinf) / (self.V0 - Vinf) )

                    XTX = np.dot(X.T, X)
                    XTX_inv = 1/XTX
                    XTY = np.dot(X.T, y)

                    b = np.dot(XTX_inv, XTY)
                    tau_est.append(b)

                    yhat = np.dot(X, b)


                else:

                    yhat = self.predict_spks(tau = force_tau, thresh = thresh, Vinf = Vinf, V0 = self.V0)

                    if np.all(np.isnan(yhat)):
                        continue

                    tau_est.append(force_tau)

                SSE.append(np.sum( (y - yhat)**2 ))

                Vinf_.append(Vinf)
                thresh_.append(thresh)

        self.tau = tau_est[np.nanargmin(SSE)]
        self.thresh = thresh_[np.nanargmin(SSE)]
        self.Vinf_est = Vinf_[np.nanargmin(SSE)]
        self.SSE_opt = np.nanmin(SSE)

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


class IASpikePredictor(OhmicSpkPredictor):

    def fit_spks(self, thresh_guesses = 'default', Vinput_guesses = 'default',
    gaprime_guesses = 'default', tauh_guesses = 'default', force_tau = None,
    sim_dt = 0.001, max_time = 10, verbose = False):

        if thresh_guesses == 'default':
            thresh_guesses = np.linspace(-50, -25, 6)

        if Vinput_guesses == 'default':
            Vinput_guesses = np.linspace(10, 100, 5)

        if gaprime_guesses == 'default':
            gaprime_guesses = np.linspace(0, 20, 5)

        if tauh_guesses == 'default':
            tauh_guesses = np.linspace(1, 3, 4)

        y = np.array(self.spks)
        SSE         = []
        tau     = []
        thresh  = []
        Vinput  = []
        gaprime = []
        tauh    = []

        for i, thresh in enumerate(thresh_guesses):

            if verbose:
                print 'Simulating {:.1f}%'.format(100* (i + 1) / len(thresh_guesses))

            for j, Vinput in enumerate(Vinput_guesses):

                # Skip combinations that will never produce spks.
                if Vinput < thresh:
                    continue

                for h_tau in tauh_guesses:

                    for gaprime in gaprime_guesses:

                        # List of spk predictions in units of tau_mem
                        x = []

                        for V0 in self.V0:
                            x.append(self.predict_spk(gaprime, thresh, V0, Vinput, h_tau = h_tau, dt = sim_dt, max_time = max_time))

                        x = np.array(x)

                        if force_tau is None:
                            # Calculate optimal tau
                            tau_ = np.sum(x * y) / np.sum(x * x)
                        else:
                            tau_ = force_tau

                        SSE.append(np.sum((y - x * tau_)**2))
                        tau.append(tau_)
                        thresh.append(thresh)
                        Vinput.append(Vinput)
                        gaprime.append(gaprime)
                        tauh.append(h_tau)

        ind = np.nanargmin(SSE)
        self.tau = tau[ind]
        self.thresh = thresh[ind]
        self.Vinput = Vinput[ind]
        self.gaprime = gaprime[ind]
        self.tauh = tauh[ind]
        self.SSE_opt = SSE[ind]

        output_dict = {
            'SSE': SSE,
            'tau': tau,
            'thresh': thresh,
            'Vinput': Vinput,
            'gaprime': gaprime
        }

        return output_dict

    @staticmethod
    def predict_spk(ga, thresh, V0, Vin, h_tau = 1.5, dt = 0.001, max_time = 10.):

        El = -60.
        Ea = -101.

        m_Vhalf = -27.
        m_k = 0.113

        h_Vhalf = -74.7
        h_k = -0.11

        # Set initial condition
        V_t = V0
        h_t = 1./ (1 + np.exp(-h_k * (V0 - h_Vhalf)))
        m_t = 1./ (1 + np.exp(-m_k * (V0 - m_Vhalf)))

        cnt = 0
        t = 0.
        while V_t < thresh and t < max_time:

            dV = -(V_t - El) - ga * m_t * h_t * (V_t - Ea) + Vin

            m_t = 1./ (1 + np.exp(-m_k * (V_t - m_Vhalf)))

            h_inf = 1./ (1 + np.exp(-h_k * (V_t - h_Vhalf)))
            dh = (h_inf - h_t) / h_tau
            h_t += dh * dt

            V_t += dV * dt

            t += dt
            cnt += 1

        return t



def _predict_spk_for_scipy(params, V0_vec):

    ga, thresh, Vin, h_tau, tau_mem = params
    thresh *= 2
    Vin *= 2
    h_tau /= 10

    dt = 0.001
    max_time = 2.

    El = -60.
    Ea = -101.

    m_Vhalf = -27.
    m_k = 0.113

    h_Vhalf = -74.7
    h_k = -0.11

    t_vec = []
    for V0 in V0_vec:

        # Set initial condition
        V_t = V0
        h_t = 1./ (1 + np.exp(-h_k * (V0 - h_Vhalf)))
        m_t = 1./ (1 + np.exp(-m_k * (V0 - m_Vhalf)))

        cnt = 0
        t = 0.
        while V_t < thresh and t < max_time:

            dV = -(V_t - El) - ga * m_t * h_t * (V_t - Ea) + Vin

            m_t = 1./ (1 + np.exp(-m_k * (V_t - m_Vhalf)))

            h_inf = 1./ (1 + np.exp(-h_k * (V_t - h_Vhalf)))
            dh = (h_inf - h_t) / h_tau
            h_t += dh * dt

            V_t += dV * dt

            t += dt
            cnt += 1

        t_vec.append(t)

    t_vec = np.array(t_vec)
    V0_vec = np.array(V0_vec)

    #tau_mem_est = np.sum(t_vec * V0_vec) / np.sum(t_vec * t_vec)

    return np.sum((V0_vec - t_vec * tau_mem)**2)
