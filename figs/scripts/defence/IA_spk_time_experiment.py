#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gs
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./analysis/spk_timing/IA_mod')
sys.path.append('./figs/scripts')
from grr import pltools
from grr.cell_class import Cell, Recording
import IAmod

#%% DEFINE NECESSARY CLASSES

class OhmicSpkPredictor(object):

    def __init__(self):
        pass

    def add_recordings(self, recs, baseline = None, ss = None, tau = None, V_channel = 0, I_channel = 1, dt = 0.1):

        """
        Recs should be a list of Recording objects with dimensionality [channel, time, sweep]
        """

        self.tau_est = None
        self.thresh_est = None
        self.V0 = None
        self.Vinf_est = None

        recs_arr = np.array(recs)
        self.V = deepcopy(recs_arr[:, V_channel, :, :])
        self.I = deepcopy(recs_arr[:, I_channel, :, :])

        self.dt = dt

        if type(recs[0]) is Recording and (baseline is not None and ss is not None):

            self.Rins = []
            self.taus = []

            for rec in recs:
                if tau is None:
                    tmp = rec.fit_test_pulse(baseline, ss, V_clamp = False, verbose = False)
                    self.Rins.append(tmp['R_input'].mean())
                else:
                    tmp = rec.fit_test_pulse(baseline, ss, tau = tau, V_clamp = False, verbose = False)
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

        self.tau_est = tau_est[np.nanargmin(SSE)]
        self.thresh_est = thresh_[np.nanargmin(SSE)]
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

        tau         = kwargs.get('tau', self.tau_est)
        thresh      = kwargs.get('thresh', self.thresh_est)
        V0          = kwargs.get('V0', self.V0)
        Vinf        = kwargs.get('Vinf', self.Vinf_est)

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
        tau_est     = []
        thresh_est  = []
        Vinput_est  = []
        gaprime_est = []
        tauh_est    = []

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
                            tau_est_ = np.sum(x * y) / np.sum(x * x)
                        else:
                            tau_est_ = force_tau

                        SSE.append(np.sum((y - x * tau_est_)**2))
                        tau_est.append(tau_est_)
                        thresh_est.append(thresh)
                        Vinput_est.append(Vinput)
                        gaprime_est.append(gaprime)
                        tauh_est.append(h_tau)

        ind = np.nanargmin(SSE)
        self.tau_est = tau_est[ind]
        self.thresh_est = thresh_est[ind]
        self.Vinput_est = Vinput_est[ind]
        self.gaprime_est = gaprime_est[ind]
        self.tauh_est = tauh_est[ind]
        self.SSE_opt = SSE[ind]

        output_dict = {
            'SSE': SSE,
            'tau_est': tau_est,
            'thresh_est': thresh_est,
            'Vinput_est': Vinput_est,
            'gaprime_est': gaprime_est
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


#%% IMPORT DATA

DATA_PATH = './data/raw/5HT/spk_time/'

"""
inventory = pd.read_csv(DATA_PATH + 'index.csv')
inventory_4AP = inventory.loc[inventory['Cell'] == 'DRN332', :]
inventory_4AP['cumcount'] = inventory_4AP.groupby('Cell').cumcount()
fnames_4AP = inventory_4AP.pivot('Cell', 'cumcount', values = 'Recording')
"""

fnames_baseline = ['18627043.abf', '18627044.abf', '18627045.abf', '18627046.abf', '18627047.abf']
fnames_4AP = ['18627053.abf', '18627054.abf', '18627055.abf']
fnames_wash = ['18627062.abf', '18627063.abf', '18627064.abf']

recs_baseline = Cell().read_ABF([DATA_PATH + fname for fname in  fnames_baseline])
recs_4AP = Cell().read_ABF([DATA_PATH + fname for fname in fnames_4AP])

with open(DATA_PATH + 'predictors_unconstrained.pyc', 'rb') as f:
    predictors = pickle.load(f)

#%% MAKE FIGURE

ga = 10
tau_h = 3
input_strength = 26
Vinput = np.empty((5000, 1))
Vinput[1000:] = input_strength

toy_spk_predictor = IASpikePredictor()
toy_IA_neuron = IAmod.IAmod(ga, tau_h, 0)
toy_IA_neuron.vreset = -60
toy_ohmic_neuron = IAmod.IAmod(0, tau_h, 0)
toy_ohmic_neuron.vreset = -60


IMG_PATH = './figs/ims/defence/'

plt.style.use('./figs/scripts/defence/defence_mplrc.dms')

spec_mod_outer = gs.GridSpec(
    1, 2,
    left = 0.05, top = 0.85, right = 0.95, bottom = 0.15,
    wspace = 0.5, width_ratios = [1, 0.75]
)
spec_model = gs.GridSpecFromSubplotSpec(2, 2, spec_mod_outer[:, 0], height_ratios = [1, 0.2], hspace = 0.1)

fig = plt.figure(figsize = (5, 2))

### A: simulated proof-of-principle

ax_ohmic = plt.subplot(spec_model[0, 1])
plt.title('Linear model', loc = 'left')
ax_ohmic_I = plt.subplot(spec_model[1, 1])

ax_IA = plt.subplot(spec_model[0, 0])
plt.title('Linear + $I_A$', loc = 'left')
ax_IA_I = plt.subplot(spec_model[1, 0])

for i, V0 in enumerate([-70, -50]):

    Vinput[:1000] = toy_IA_neuron.ss_clamp(V0)
    V_mat_IA, spks_mat, _, _ = toy_IA_neuron.simulate(V0, Vinput)
    V_mat_IA[spks_mat] = 0
    ax_IA.plot(V_mat_IA, 'b-', linewidth = 0.5, alpha = 1/(i + 1))
    ax_IA_I.plot(Vinput, color = 'gray', linewidth = 0.5, alpha = 1/(i + 1))

    Vinput[:1000] = toy_ohmic_neuron.ss_clamp(V0)
    V_mat_ohmic, spks_mat, _, _ = toy_ohmic_neuron.simulate(V0, Vinput)
    V_mat_ohmic[spks_mat] = 0
    ax_ohmic.plot(V_mat_ohmic, 'k-', linewidth = 0.5, alpha = 1/(i + 1))
    ax_ohmic_I.plot(Vinput, color = 'gray', linewidth = 0.5, alpha = 1/(i + 1))

ax_IA.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
ax_IA.annotate('-70mV', (5000, -68), ha = 'right')

ax_ohmic.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
ax_ohmic.annotate('-70mV', (5000, -68), ha = 'right')

ax_IA_I.annotate('26mV', (5000, 24), ha = 'right', va = 'top')
ax_ohmic_I.annotate('26mV', (5000, 24), ha = 'right', va = 'top')

#pltools.add_scalebar(ax = ax_IA, y_units = 'mV', omit_x = True, anchor = (-0.05, 0), y_label_space = (-0.05))
pltools.hide_border(ax = ax_IA)
pltools.hide_ticks(ax = ax_IA)
pltools.hide_border(ax = ax_IA_I)
pltools.hide_ticks(ax = ax_IA_I)

#pltools.add_scalebar(ax = ax_ohmic, y_units = 'mV', omit_x = True, anchor = (-0.05, 0), y_label_space = (-0.05))
pltools.hide_border(ax = ax_ohmic)
pltools.hide_ticks(ax = ax_ohmic)
pltools.hide_border(ax = ax_ohmic_I)
pltools.hide_ticks(ax = ax_ohmic_I)


plt.subplot(spec_mod_outer[:, 1])
#plt.title('\\textbf{{A3}} Effect on spike latency', loc = 'left')
V0_vec = np.linspace(-90, -45)
IA_spk_times = []
ohmic_spk_times = []
for V0 in V0_vec:
    IA_spk_times.append(
        toy_spk_predictor.predict_spk(ga, -45, V0, input_strength, 3, max_time = 10)
    )
    ohmic_spk_times.append(
        toy_spk_predictor.predict_spk(0, -45, V0, input_strength, 3, max_time = 10)
    )

plt.plot(V0_vec, ohmic_spk_times, 'k-', label = 'Linear model')
plt.plot(V0_vec, IA_spk_times, 'b-', label = 'Linear + $I_A$')
plt.ylabel('Spike latency $\\tau_{{\mathrm{{mem}}}}$')
plt.xlabel('$V_{{\mathrm{{pre}}}}$ (mV)')
plt.legend()
pltools.hide_border('tr')


if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_experiment_predictions.png', dpi = 300)

plt.show()


#%%

spec_cell_outer = gs.GridSpec(
    1, 2,
    left = 0.05, top = 0.85, right = 0.95, bottom = 0.2,
    wspace = 0.5, width_ratios = [1, 0.75]
)
spec_4AP = gs.GridSpecFromSubplotSpec(2, 2, spec_cell_outer[:, 0], height_ratios = [1, 0.2], hspace = 0.1)

plt.figure(figsize = (5, 2))

### B: real neurons

trace_time_slice = slice(25400, 28400)
t_vec = np.arange(0, 300, 0.1)
V_ax_bl = plt.subplot(spec_4AP[0, 0])
plt.title(' 5HT neuron', loc = 'left')
plt.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
plt.annotate('-70mV', (300, -72), ha = 'right', va = 'top')
plt.annotate('', (140, 9), (240, 9), arrowprops = {'arrowstyle': '<->'})
plt.text(190, 14, '$\\Delta t_\\mathrm{{spk}}$', ha = 'center')
plt.ylim(-85, 40)

I_ax_bl = plt.subplot(spec_4AP[1, 0])
plt.annotate('30pA', (300, 28), ha = 'right', va = 'top')
pltools.hide_border()
pltools.hide_ticks()

for i, sweep_no in enumerate([3, 8]):
    V_ax_bl.plot(
        t_vec, recs_baseline[0][0, trace_time_slice, sweep_no],
        'b-', lw = 0.5, alpha = 1/(i + 1)
    )
    I_ax_bl.plot(
        t_vec, recs_baseline[0][1, trace_time_slice, sweep_no],
        color = 'gray', lw = 0.5, alpha = 1/(i + 1)
    )

pltools.add_scalebar(
    y_units = 'mV', x_units = 'ms', anchor = (-0.15, 0.5),
    y_label_space = (-0.05), x_on_left = False, x_size = 50,
    bar_space = 0, ax = V_ax_bl
)

V_ax_4AP = plt.subplot(spec_4AP[0, 1])
plt.title('$I_A$ blocked', loc = 'left')
plt.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
plt.annotate('-70mV', (300, -72), ha = 'right', va = 'top')
plt.annotate('\\textbf{{+4AP}}', (50, 25), ha = 'center')
plt.annotate('', (120, 9), (210, 9), arrowprops = {'arrowstyle': '<->'})
plt.text(165, 14, '$\\Delta t_\\mathrm{{spk}}$', ha = 'center')
plt.ylim(-85, 40)

I_ax_4AP = plt.subplot(spec_4AP[1, 1])
plt.annotate('30pA', (300, 28), ha = 'right', va = 'top')

for i, sweep_no in enumerate([4, 11]):
    V_ax_4AP.plot(
        t_vec, recs_4AP[0][0, trace_time_slice, sweep_no],
        'k-', lw = 0.5, alpha = 1/(i + 1)
    )
    I_ax_4AP.plot(
        t_vec, recs_4AP[0][1, trace_time_slice, sweep_no],
        color = 'gray', lw = 0.5, alpha = 1/(i + 1)
    )

pltools.hide_border(ax = V_ax_4AP)
pltools.hide_ticks(ax = V_ax_4AP)
pltools.hide_border(ax = I_ax_4AP)
pltools.hide_ticks(ax = I_ax_4AP)



latency_dist_ax = plt.subplot(spec_cell_outer[:, 1])
#plt.title('\\textbf{{B3}} Sample latency distribution', loc = 'left')

ex_predictor = predictors[4]['pred_IA']

plt.plot(ex_predictor.V0, ex_predictor.spks, 'k.')

V0_vec_ex = np.linspace(-95, -40)
IA_spk_times_ex = []
for V0 in V0_vec_ex:
    IA_spk_times_ex.append(ex_predictor.predict_spk(
        ex_predictor.gaprime_est, ex_predictor.thresh_est, V0,
        ex_predictor.Vinput_est, ex_predictor.tauh_est
    ))

plt.plot(
    V0_vec_ex, np.array(IA_spk_times_ex) * ex_predictor.tau_est,
    'b--', label = 'Linear + $I_A$ fit'
)
plt.annotate('\\textbf{{1}}', (-69, 140))
plt.annotate('\\textbf{{2}}', (-50, 25))
plt.ylabel('Spike latency (ms)')
plt.xlabel('$V_{{\mathrm{{pre}}}}$ (mV)')
plt.legend(loc = 'upper right')
pltools.hide_border('tr')
plt.ylim(-10, 190)

bbox_anchor = (0.05, 0.05, 0.5, 0.4)
ins2 = inset_axes(
    latency_dist_ax, '40%', '100%', loc = 'center right',
    bbox_to_anchor = bbox_anchor, bbox_transform = latency_dist_ax.transAxes
)
ins1 = inset_axes(
    latency_dist_ax, '40%', '100%', loc = 'center left',
    bbox_to_anchor = bbox_anchor, bbox_transform = latency_dist_ax.transAxes
)


ins1.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
ins1.plot(np.arange(0, 195, 0.1), ex_predictor.V[0, 26250:28200, 5], 'k-', lw = 0.5)
ins1.annotate('\\textbf{{1}}', (0, 60))
pltools.hide_border(ax = ins1)
pltools.hide_ticks(ax = ins1)
ins1.set_ylim(-70, 60)

ins2.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
ins2.plot(np.arange(0, 195, 0.1), ex_predictor.V[0, 26250:28200, 8], 'k-', lw = 0.5)
ins2.annotate('\\textbf{{2}}', (0, 60))
pltools.hide_border(ax = ins2)
pltools.hide_ticks(ax = ins2)
ins2.set_ylim(-70, 60)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_experiment_observed.png', dpi = 300)

plt.show()
