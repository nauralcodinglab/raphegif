from __future__ import division

from copy import deepcopy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import numba as nb

from grr.cell_class import Cell, Recording


def param_str(name, value, units=None):
    string_ = '{name:<12}: {value:>9.3f} {units}'.format(
        name=name, value=np.mean(value), units=units
    )
    return string_

class LatencyModel(object):
    """Base class for spike latency models.

    Spike latency models explicitly model the latency to the first spike
    after the onset of a square input pulse.

    Attributes
    ----------
    dt : float
        Time step (ms).
    tau_raw, Rin_raw : float array-like or None
        Membrane parameters estimated from recordings.
    V_raw, I_raw : float array-like or None
        Membrane voltage and current from recordings.
    t_vec, t_mat : float array-like or None
        Time support arrays for membrane voltage and current.
    spks_raw : float array-like or None
        Spike times scraped from recording.
    V0_raw, Vin_raw : float array-like or None
        Membrane voltage before and after current step (Vin_raw computed based
        on Rin_raw and I_raw).
    tau_est, Vin_est, thresh_est : float or None
        Parameters fitted to data.

    Methods
    -------
    add_recordings
        Store recording objects for fitting.
    scrape_data
        Scrape summary statistics from `V` attribute.
    fit
        Estimate model parameters to match spike latencies in data.
    predict
        Predict spike latencies given model.

    """

    def __init__(self, dt=0.1):
        """Initialize LatencyModel."""
        self.dt = dt

        # Data attributes.
        self.V_raw = None
        self.I_raw = None

        # Attributes scraped from data.
        self.tau_raw = None
        self.Rin_raw = None
        self.spks_raw = None
        self.V0_raw = None
        self.Vin_raw = None

        # Parameters fitted to data.
        self.tau_est = None
        self.Vin_est = None
        self.thresh_est = None

    def __str__(self):
        str_attrs = [
            'tau_raw', 'Rin_raw', 'V0_raw', 'Vin_raw', 'tau_est',
            'Vin_est', 'thresh_est'
        ]
        model_description = [
            param_str(attr, getattr(self, attr)) for attr in str_attrs
        ]
        return '\n'.join(model_description)

    @property
    def t_vec(self):
        """Time support vector."""
        if self.V_raw is None:
            return None
        else:
            return np.arange(0, (self.V_raw.shape[1] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self, reps='default'):
        """Time support matrix."""
        if self.V_raw is None:
            return None
        else:
            if reps == 'default':
                reps = self.V_raw.shape[2]
            return np.tile(self.t_vec[:, np.newaxis], (1, reps))

    def add_recordings(
        self, recs, baseline=None, ss=None, tau=None,
        V_chan=0, I_chan=1, plot_tau=False
    ):
        """Ingest recordings into LatencyModel.

        Arguments
        ---------
        recs : list of `Recording` objects
        baseline, ss : pair of ints
            Time indices to average for baseline and steady-state of test pulse.
        tau : triple of ints
            Parameters for time constant fitting to test pulse (see
            `cell_class.Recording.fit_test_pulse`).
        V_chan, I_chan : int
            Indices of V and I channels in `Recording` objects.

        """
        recs_arr = np.array(recs)
        self.V_raw = deepcopy(recs_arr[:, V_chan, :, :])
        self.I_raw = deepcopy(recs_arr[:, I_chan, :, :])

        if (
            all([isinstance(rec, Recording) for rec in recs])
            and (baseline is not None and ss is not None)
        ):
            Rin_tmp = []
            if tau is None:
                # Get Rin but not tau.
                for rec in recs:
                    tmp = rec.fit_test_pulse(
                        baseline, ss, V_clamp=False, verbose=False,
                        V_chan=V_chan, I_chan=I_chan
                    )
                    Rin_tmp.append(tmp['R_input'].mean())
            else:
                # Get Rin and tau.
                tau_tmp = []
                for rec in recs:
                    tmp = rec.fit_test_pulse(
                        baseline, ss, tau=tau, V_clamp=False,
                        V_chan=V_chan, I_chan=I_chan,
                        verbose=False, plot_tau=plot_tau
                    )
                    Rin_tmp.append(tmp['R_input'].mean())
                    tau_tmp.append(tmp['tau'])
                self.tau_raw = np.array(tau_tmp)
            self.Rin_raw = np.array(Rin_tmp)

    def scrape_data(
        self,
        V0_range=(2495, 2595), Vin_range=(2750, 2800), baseline_range=(0, 100),
        quiescent_until=2647, dVdt_thresh=10, exclude_above=-55
    ):
        """Scrape spks, V0, and Vin from attached V and I.

        Places all spk times together in a big honking list. Complementary big
        honking lists of V0 and Vin are generated. Only takes the first spk
        after `quiescent_until` from each sweep.

        Arguments
        ---------
        V0_range : tuple of two floats
            Time (ms) range from which to get V0
        Vin_range : tuple of two floats
            Time (ms) range from which to calculate Vin *based on I_probe*
        quiescent_until : float
            Discard sweeps that have spikes before this point (ms).
        dVdt_thresh : float
            Voltage derivative threshold to use for spk detection (mV/ms).

        """
        V0_slice = slice(int(V0_range[0] / self.dt), int(V0_range[1] / self.dt))
        Vin_slice = slice(int(Vin_range[0] / self.dt), int(Vin_range[1] / self.dt))
        baseline_slice = slice(int(baseline_range[0] / self.dt), int(baseline_range[1] / self.dt))

        dVdt = np.gradient(self.V_raw, axis=1) / self.dt

        spks = []
        V0 = []
        Vin = []

        for rec_ind in range(dVdt.shape[0]):

            V_baseline = self.V_raw[:, baseline_slice, :].mean()
            I_baseline = self.I_raw[:, baseline_slice, :].mean()

            for sw_ind in range(dVdt.shape[2]):

                # Extract spikes and V0.
                spks_i = np.where(dVdt[rec_ind, :, sw_ind] > dVdt_thresh)[0] * self.dt
                V0_i = self.V_raw[rec_ind, V0_slice, sw_ind].mean()

                # Apply exclusion criteria.
                if (
                    any(spks_i < quiescent_until)
                    or len(spks_i) < 1
                    or V0_i > exclude_above
                ):
                    continue

                # Extract Vin
                I_probe = self.I_raw[rec_ind, Vin_slice, sw_ind].mean()
                Vin_i = self.Rin_raw[rec_ind] * (I_probe - I_baseline) * 1e-3 + V_baseline

                # Assign output
                spks.append(spks_i[0] - quiescent_until)
                V0.append(V0_i)
                Vin.append(Vin_i)

        # Store in class attributes
        self.spks_raw = np.array(spks)
        self.V0_raw = np.array(V0)
        self.Vin_raw = np.array(Vin)

    def fit(self):
        """Fit LatencyModel to data (implemented by derived classes)."""
        raise NotImplementedError(
            '`LatencyModel.fit` method must be implemented by derived classes.'
        )

    def predict(self):
        """Predict spikes (implemented by derived classes)."""
        raise NotImplementedError(
            '`LatencyModel.predict` method must be implemented by derived '
            'classes.'
        )


class OhmicLatencyModel(LatencyModel):

    def __init__(self, dt=0.1):
        """Initialize OhmicLatencyModel."""
        super(OhmicLatencyModel, self).__init__(dt)

    def __str__(self):
        """Summary of OhmicLatencyModel."""
        return super(OhmicLatencyModel, self).__str__()

    def fit(self, thresh_guesses, Vin_guesses, force_tau=None, verbose=False):
        """Fit OhmicLatencyModel to attached data.

        Perform a grid search over possible threshold and Vin values, finding
        the optimal membrane time constant for each combination of parameters.

        Arguments
        ---------
        thresh_guesses, Vin_guesses : float array-like
            Values to try during parameter search.
        force_tau : float, `raw`, or None
            Constrain model to a given membrane time constant. Set to `raw` to
            use tau estimated from raw data or `None` to fit tau based on
            latency.
        verbose : bool
            Print information about progress.

        """
        if force_tau == 'raw':
            force_tau = self.tau_raw

        tau_est_tmp = []
        SSE_tmp = []

        y = np.asarray(self.spks_raw)
        param_grid = ParameterGrid({
            'thresh': thresh_guesses,
            'Vin': Vin_guesses
        })
        for i, param_set in enumerate(param_grid):
            if verbose:
                print '\rFitting {:.1f}%'.format(100 * i / len(param_grid)),

            # Skip parameter combinations that can never spike.
            if param_set['Vin'] < param_set['thresh']:
                tau_est_tmp.append(np.nan)
                SSE_tmp.append(np.nan)
                continue

            if force_tau is None:
                X = - np.log(
                    (param_set['thresh'] - param_set['Vin'])
                    / (self.V0_raw - param_set['Vin'])
                )

                XTX = np.dot(X.T, X)
                XTX_inv = 1 / XTX
                XTY = np.dot(X.T, y)
                b = np.dot(XTX_inv, XTY)
                tau_est_tmp.append(b)

                yhat = np.dot(X, b)

            else:
                yhat = self.predict(
                    tau=force_tau, thresh=param_set['thresh'],
                    Vin=param_set['Vin'], V0=self.V0_raw
                )
                if np.all(np.isnan(yhat)):
                    # Skip if no spikes are produced.
                    tau_est_tmp.append(np.nan)
                    SSE_tmp.append(np.nan)
                    continue
                else:
                    tau_est_tmp.append(force_tau)

            SSE_tmp.append(np.sum((y - yhat)**2))

        if all(np.isnan(SSE_tmp)):
            warnings.warn(
                'All NaN loss in `OhmicLatencyModel.fit`. Abandoning fit '
                'without storing parameter estimates.'
            )
            return None

        self.tau_est = tau_est_tmp[np.nanargmin(SSE_tmp)]
        self.SSE_opt = np.nanmin(SSE_tmp)

        opt_params = param_grid[np.nanargmin(SSE_tmp)]
        self.thresh_est = opt_params['thresh']
        self.Vin_est = opt_params['Vin']

    def predict(self, tau=None, thresh=None, V0=None, Vin=None):
        """Predict spike latencies.

        Arguments
        ---------
        tau, thresh, V0, Vin : float, array-like, or None
            Parameters to use for predictive model. Set to `None` to use
            fitted/scraped values.

        Returns
        -------
        Predicted spike latency (or latencies).

        """
        if tau is None:
            tau = self.tau_est
        if thresh is None:
            thresh = self.thresh_est
        if V0 is None:
            V0 = self.V0_raw
        if Vin is None:
            Vin = self.Vin_est

        latency_predictions = -tau * np.log((thresh - Vin) / (V0 - Vin))
        return latency_predictions


class IALatencyModel(LatencyModel):

    def __init__(self, dt=0.1):
        """Initialize IALatencyModel."""
        super(IALatencyModel, self).__init__(dt)

        # Initialize IALatencyModel-specific attributes.
        self.ga_est = None
        self.tauh_est = None

    def __str__(self):
        model_description = [super(IALatencyModel, self).__str__()]
        str_attrs = ['ga_est', 'tauh_est']
        model_description.extend(
            [param_str(attr, getattr(self, attr)) for attr in str_attrs]
        )
        return '\n'.join(model_description)

    def fit(
        self, thresh_guesses, Vin_guesses, ga_guesses, tauh_guesses,
        force_tau=None, max_time=10., time_step=1e-2, verbose=False
    ):
        """Fit IALatencyModel to attached data.

        Perform a grid search over possible threshold, Vin, ga, and tauh
        values, finding the optimal membrane time constant for each set of
        parameters.

        Arguments
        ---------
        thresh_guesses, Vin_guesses, ga_guesses, tau_guesses : float array-like
            Values to try during parameter search.
        force_tau : float, `raw`, or None
            Constrain model to a given membrane time constant. Set to `raw` to
            use tau estimated from raw data or `None` to fit tau based on
            latency.
        max_time : float
            Maximum length of numerically integrated voltage.
        verbose : bool
            Print information about progress.

        """
        if force_tau == 'raw':
            force_tau = self.tau_raw

        tau_est_tmp = []
        SSE_tmp = []

        y = np.asarray(self.spks_raw)

        param_grid = ParameterGrid({
            'thresh': thresh_guesses,
            'Vin': Vin_guesses,
            'ga': ga_guesses,
            'tauh': tauh_guesses,
        })
        for i, param_set in enumerate(param_grid):
            if verbose:
                print '\rFitting {:.1f}%'.format(100 * i / len(param_grid)),

            # Skip parameter combinations that will never spike.
            if param_set['Vin'] < param_set['thresh']:
                tau_est_tmp.append(np.nan)
                SSE_tmp.append(np.nan)
                continue

            # List of latency_predictions in units of tau_mem.
            latency_predictions = self.predict(
                1., param_set['thresh'], self.V0_raw, param_set['Vin'],
                param_set['ga'], param_set['tauh'], max_time, time_step
            )

            if force_tau is None:
                # Find optimal tau.
                tau_tmp = (
                    np.sum(latency_predictions * y)
                    / np.sum(latency_predictions**2)
                )
            else:
                tau_tmp = force_tau

            # Store tau and loss.
            tau_est_tmp.append(tau_tmp)
            SSE_tmp.append(np.sum((y - latency_predictions * tau_tmp)**2))

        if all(np.isnan(SSE_tmp)):
            warnings.warn(
                'All NaN loss in `OhmicLatencyModel.fit`. Abandoning fit '
                'without storing parameter estimates.'
            )
            return None

        # Store parameters.
        self.tau_est = tau_est_tmp[np.nanargmin(SSE_tmp)]
        self.SSE_opt = np.nanmin(SSE_tmp)

        opt_params = param_grid[np.nanargmin(SSE_tmp)]
        self.thresh_est = opt_params['thresh']
        self.Vin_est = opt_params['Vin']
        self.ga_est = opt_params['ga']
        self.tauh_est = opt_params['tauh']

    def predict(
        self,
        tau=None, thresh=None, V0=None, Vin=None,
        ga=None, tauh=None, max_time=np.inf
    ):
        """Predict spike latencies.

        Arguments
        ---------
        tau, thresh, ..., tauh : float, array-like, or None
            Parameters to use for predictive model. Set to `None` to use
            fitted/scraped values.

        Returns
        -------
        Predicted spike latency (or latencies).

        """
        # Broadcast inputs to same shape.
        argdict = {
            'tau': self.tau_est if tau is None else tau,
            'thresh': self.thresh_est if thresh is None else thresh,
            'V0': self.V0_raw if V0 is None else V0,
            'Vin': self.Vin_est if Vin is None else Vin,
            'ga': self.ga_est if ga is None else ga,
            'tauh': self.tauh_est if tauh is None else tauh,
        }
        array_arglens = []
        for arg in argdict:
            argdict[arg] = np.atleast_1d(
                np.array(argdict[arg], dtype=np.float64)
            )
            if len(argdict[arg]) > 1:
                array_arglens.append(len(argdict[arg]))
        if (
            len(array_arglens) > 1
            and not all([len_ == max(array_arglens) for len_ in array_arglens])
        ):
            raise ValueError(
                'operands could not be broadcasted to same shape.'
            )
        # Do broadcasting.
        if len(array_arglens) > 0:
            for arg in argdict:
                argdict[arg] = np.broadcast_to(
                    argdict[arg], max(array_arglens)
                )

        # Genreate latency predictions using accelerated private function.
        latency_predictions = []
        for i in range(len(argdict['tau'])):
            latency_predictions.append(
                self._timed_integrate_to_bound(
                    argdict['V0'][i], argdict['Vin'][i], argdict['thresh'][i],
                    argdict['tau'][i], argdict['ga'][i], argdict['tauh'][i],
                    np.float64(max_time), np.float64(self.dt)
                )
            )

        return np.asarray(latency_predictions)

    @staticmethod
    @nb.jit(
        nb.float64(
            nb.float64, nb.float64, nb.float64, nb.float64,
            nb.float64, nb.float64, nb.float64, nb.float64
        ),
        cache=True,
        nopython=True
    )
    def _timed_integrate_to_bound(
        V0, Vin, thresh, tau, ga, tauh, max_time, time_step
    ):
        """Find time to integrate voltage of IALatencyModel up to thresh.

        Arguments
        ---------
        V0 : double
            Initial voltage (mV).
        Vin : double
            Voltage input (mV).
        thresh : double
            Bound at which to stop integration (mV).
        tau : double
            Membrane time constant (ms).
        ga : double
            Relative A-type conductance (ga/gl).
        tauh : double
            Relative inactivation time constant of IA (tauh/tau_membrane).
        max_time : double
            Time at which to stop integration if bound is not reached (tau).
        time_step : double
            Time step of numerical integration (tau).

        Returns
        -------
        t : double
            Time until bound (or max_time; ms).

        """
        raise RuntimeError('ga should be relative ga (ga/gl)')
        # Unit conversions.
        effective_tauh = tauh / tau
        effective_time_step = time_step / tau

        # Model hyperparameters.
        El = -60.
        Ea = -101.
        m_Vhalf = -23.7
        m_k = 0.0985
        m_A = 1.61

        h_Vhalf = -59.2
        h_k = -0.165
        h_A = 1.03

        # Set initial condition
        V_t = V0
        h_t = h_A / (1 + np.exp(-h_k * (V0 - h_Vhalf)))
        m_t = m_A / (1 + np.exp(-m_k * (V0 - m_Vhalf)))

        # Integrate over time.
        cnt = 0
        while V_t < thresh and cnt * time_step < max_time:

            # Integrate voltage using values form last time step.
            dV = -(V_t - El) - ga * m_t * h_t * (V_t - Ea) + Vin - V0  # Should be rel ga!!

            # Update state variables.
            m_t = m_A / (1 + np.exp(-m_k * (V_t - m_Vhalf)))
            h_inf = h_A / (1 + np.exp(-h_k * (V_t - h_Vhalf)))
            dh = (h_inf - h_t) / effective_tauh
            h_t += dh * effective_time_step

            # Update voltage for next time step.
            V_t += dV * effective_time_step

            # Increment time elapsed.
            cnt += 1

        t = np.float64(cnt) * time_step

        # Return time that V_t crosses thresh (or max_time)
        return t
