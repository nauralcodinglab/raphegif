#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import scipy.optimize as optimize

from grr.cell_class import Cell
from grr import pltools


#%% LOAD DATA

# Filenames of 20Hz train recordings.
train_fnames = ['18o25000.abf',
                '18o25005.abf',
                '18o25022.abf']

IV_fnames = ['18o25002.abf',
             '18o25006.abf',
             '18o25024.abf']

DATA_PATH = './data/raw/5HT/GABA_synapses/'

train_recs = Cell().read_ABF([DATA_PATH + fname for fname in train_fnames])
IV_recs = Cell().read_ABF([DATA_PATH + fname for fname in IV_fnames])


#%% INSPECT TRAIN RECORDINGS

stim_slice = slice(28000, 33500)
t_vec = np.arange(0, (stim_slice.stop - stim_slice.start) * 0.1, 0.1)

for tr in train_recs:

    sub = np.copy(tr[0, stim_slice, :])
    sub -= sub[:5].mean(axis = 0)

    plt.figure()
    plt.subplot(111)
    plt.title('Mean synaptic events -- +60mV, 20Hz stim, {} sweeps'.format(sub.shape[1]))
    plt.plot(t_vec, sub.mean(axis = 1), 'k-')
    plt.ylim(-20, 110)
    plt.ylabel('I (pA)')
    plt.xlabel('Time (ms)')
    plt.show()


#%% INSPECT IV CURVE RECORDINGS

stim_slice = slice(7400, 8600)
t_vec = np.arange(0, (stim_slice.stop - stim_slice.start) * 0.1, 0.1)

for tr in IV_recs:

    sub_I = np.copy(tr[0, stim_slice, :])
    sub_I -= sub_I[:5].mean(axis = 0)

    sub_V = np.copy(tr[1, stim_slice, :])
    sub_V = sub_V.mean(axis = 0)

    plt.figure()
    plt.subplot(121)
    plt.title('Sample traces')
    plt.plot(t_vec, sub, 'k-', lw = 0.5, alpha = 0.7)
    plt.ylim(-100, 200)

    plt.subplot(122)
    plt.title('Voltage')
    plt.plot(sub_V)
    plt.xlabel('Sweep no.')

    plt.show()

plt.hist(tr[1, stim_slice, :].std(axis = 0))

#%% DEFINE FUNCTIONS

def bin_and_average(rec, start, stop, SD_thresh = 4, min_sweeps = 5, verbose = True):

    """
    Collect sweeps at approximately equal voltage and average them.
    Returns tuple of averaged current traces and voltage centroids.
    """

    I_channel = 0
    V_channel = 1

    time_slice = slice(start, stop)

    # Make local copy of rec
    rec_ = np.copy(rec)
    I_arr = rec_[I_channel, time_slice, :]

    # Pre-processing of V command
    V_means = rec_[V_channel, time_slice, :].mean(axis = 0)
    V_noise_SD = rec_[V_channel, time_slice, :].std(axis = 0).mean()
    V_thresh = V_noise_SD * SD_thresh

    assert I_arr.shape[1] == len(V_means)

    # Create lists to hold output
    V_centroids = []
    I_means = []

    # Iterate over input, collecting sweeps at similar voltages
    # and removing them from input arrays.
    while len(V_means) > 0:
        Vm0 = V_means[0]
        mask = np.logical_and(V_means > (Vm0 - V_thresh), V_means < (Vm0 + V_thresh))

        V_centroid_tmp = V_means[mask].mean()
        V_means = V_means[~mask]

        I_mean_tmp = I_arr[:, mask].mean(axis = 1)
        I_arr = I_arr[:, ~mask]

        if mask.sum() >= min_sweeps:
            V_centroids.append(V_centroid_tmp)
            I_means.append(I_mean_tmp)
        else:
            if verbose:
                print('Too few sweeps (N = {}) at V = {:.2f}. Discarding.'.format(
                        mask.sum(), V_centroid_tmp
                    )
                )
            continue

    return np.array(I_means).T, np.array(V_centroids)

# Curve fitting
def exponential_curve(p, t):
    """Three parameter exponential.

    I = (A + C) * exp (-t/tau) + C

    p = [A, C, tau]
    """

    A       = p[0]
    C       = p[1]
    tau     = p[2]

    return (A + C) * np.exp(-t/tau) + C

def compute_residuals(p, func, Y, X):
    """Compute residuals of a fitted curve.

    Inputs:
        p       -- vector of function parameters
        func    -- a callable function
        Y       -- real values
        X       -- vector of points on which to compute fitted values

    Returns:
        Array of residuals.
    """

    if len(Y) != len(X):
        raise ValueError('Y and X must be of the same length.')

    Y_hat = func(p, X)

    return Y - Y_hat

def exponential_optimizer_wrapper(I, p0, dt = 0.1):

    t = np.arange(0, len(I) * dt, dt)[:len(I)]

    p = optimize.least_squares(compute_residuals, p0, kwargs = {
    'func': exponential_curve,
    'X': t,
    'Y': I
    })['x']

    no_pts = 500

    fitted_points = np.empty((2, no_pts))
    fitted_points[1, :] = np.linspace(t[0], t[-1], no_pts)
    fitted_points[0, :] = exponential_curve(p, fitted_points[1, :])

    return p, fitted_points


#%% EXTRACT IV TRACES

averaged_traces = {
    'traces': [],
    't_mats': [],
    'voltages': []
}

for tr in IV_recs:

    traces, voltages = bin_and_average(tr, 7400, 8600)

    traces -= traces[:5, :].mean(axis = 0)
    t_mat = np.tile(np.arange(0, traces.shape[0] * 0.1, 0.1)[:, np.newaxis], (1, traces.shape[1]))

    averaged_traces['traces'].append(traces)
    averaged_traces['t_mats'].append(t_mat)
    averaged_traces['voltages'].append(voltages)

    plt.figure()
    plt.plot(t_mat, traces, 'k-')
    plt.ylim(-50, 175)
    plt.show()


#%% EXTRACT DECAY AT +60mV

averaged_traces['decay'] = []
V_target = 60
fit_range = [slice(350, 1000), slice(300, 1000), slice(300, 1000)]
p0 = (125, 0, 50)

# Placeholder
sample_tr = {
    'ind': 2,
    'x_fit': None,
    'y_fit': None,
    'x_tr': None,
    'y_tr': None
}

for i in range(len(averaged_traces['traces'])):

    # Grab trace closest to target voltage
    ind = np.argmin(np.abs(averaged_traces['voltages'][i] - V_target))

    p, fitted_pts = exponential_optimizer_wrapper(
        averaged_traces['traces'][i][fit_range[i], ind], p0
    )

    averaged_traces['decay'].append(p[2])

    plt.figure()
    plt.plot(
        averaged_traces['t_mats'][i][:, ind], averaged_traces['traces'][i][:, ind],
        'k-', lw = 0.5
    )
    plt.plot(
        np.linspace(fit_range[i].start / 10, fit_range[i].stop / 10, fitted_pts.shape[1]),
        fitted_pts[0, :],
        'b--'
    )
    plt.ylim(-50, 175)
    plt.show()

    if i == sample_tr['ind']:
        sample_tr['x_fit'] = np.linspace(fit_range[i].start / 10, fit_range[i].stop / 10, fitted_pts.shape[1])
        sample_tr['y_fit'] = fitted_pts[0, :]

        sample_tr['x_tr'] = averaged_traces['t_mats'][i][:, ind]
        sample_tr['y_tr'] = averaged_traces['traces'][i][:, ind]

    print('Decay tau = {:.2f}ms at V = {:.2f}mV'.format(
        p[2], averaged_traces['voltages'][i][ind]
    ))

#%% DUMP DATA

SAMPLE_TR_PATH = './data/raw/5HT/GABA_synapses/sample_traces/'

with open(SAMPLE_TR_PATH + 'averaged_traces.pyc', 'wb') as f:
    pickle.dump(averaged_traces, f)

with open(SAMPLE_TR_PATH + 'sample_decay_fit.pyc', 'wb') as f:
    pickle.dump(sample_tr, f)


#%% MAKE NICE FIGURE

spec_outer = gs.GridSpec(2, 3, wspace = 0.4)
spec_tau = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[1, 2], width_ratios = [1, 0.4], wspace = 1)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
plt.rc('font', size = 8)

plt.figure(figsize = (6, 4))

plt.subplot(spec_outer[1, 0])
plt.title('\\textbf{{B1}} Sample eIPSCs', loc = 'left')
plt.plot(
    averaged_traces['t_mats'][sample_tr['ind']][:, :-2],
    averaged_traces['traces'][sample_tr['ind']][:, :-2],
    'k-', lw = 0.5
)
plt.ylim(-50, 175)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', x_size = 20, y_size = 50, bar_space = 0,
    x_on_left = False, anchor = (0.7, 0.6)
)

plt.subplot(spec_outer[1, 1])
plt.title('\\textbf{{B2}} I/V curves', loc = 'left')
plt.axhline(0, color = 'k', lw = 0.5)
plt.axvline(0, color = 'k', lw = 0.5)
for i in range(len(averaged_traces['t_mats'])):
    mask = 0 < np.gradient(averaged_traces['voltages'][i])
    plt.plot(
        averaged_traces['voltages'][i][mask],
        averaged_traces['traces'][i][250, mask],
        'k-'
    )
plt.xlabel('$V$ (mV)')
plt.ylabel('$I$ (pA)')

plt.subplot(spec_tau[:, 0])
plt.title('\\textbf{{B3}} $\\tau$ fit', loc = 'left')
plt.plot(sample_tr['x_tr'], sample_tr['y_tr'], 'k-', lw = 0.5)
plt.plot(sample_tr['x_fit'], sample_tr['y_fit'], 'g--')
plt.ylim(-10, 175)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', x_size = 20, y_size = 50, bar_space = 0,
    x_on_left = False, anchor = (0.7, 0.6)
)

plt.subplot(spec_tau[:, 1])
plt.title('\\textbf{{B4}}', loc = 'left')
plt.ylim(0, 35)
sns.swarmplot(y = averaged_traces['decay'], color = 'g', edgecolor = 'gray')
plt.xticks([])
plt.ylabel('$\\tau$ (ms)')
pltools.hide_border('trb')

plt.show()
