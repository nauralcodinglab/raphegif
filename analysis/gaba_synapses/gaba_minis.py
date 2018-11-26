#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import scipy.optimize as optimize
import scipy.signal as signal

import sys
sys.path.append('./analysis/gating')
sys.path.append('./figs/scripts')
from cell_class import Cell
import pltools


#%% LOAD DATA

mIPSC_fnames = ['18o23003.abf',
                '18o23004.abf',
                '18o23005.abf']

DATA_PATH = './data/GABA_synapses/'

mini_recs = Cell().read_ABF([DATA_PATH + fname for fname in mIPSC_fnames])


#%%

class MiniDetector(object):

    def __init__(self, rec, clean_data = True, baseline_range = (4500, 5000),
        remove_n_sweeps = 0, I_channel = 0, dt = 0.1):

        self.rec = rec
        self.dt = dt

        if clean_data:
            self.I = self._clean_data(rec, baseline_range, remove_n_sweeps, I_channel)

    @staticmethod
    def _clean_data(rec, baseline_range, remove_n_sweeps, I_channel):
        """
        Clean up Recording object containing minis.

        Inputs:
            baseline_range -- tuple of ints
                Range to use for baseline subtraction in timesteps.
                Discards points before the end of the baseline.
            remove_n_sweeps -- int
                Dicard this may sweeps from the beginning of the recording.
                Set to zero to discard none.
            I_channel -- int
                Current channel of the Recording to clean.

        Returns a matrix with the cleaned I channel.
        """

        baseline_slice = slice(baseline_range[0], baseline_range[1])

        I = rec[I_channel, :, :] - rec[I_channel, baseline_slice, :].mean(axis = 0)
        I_sub = I[baseline_range[1]:, remove_n_sweeps:]

        return I_sub

    @staticmethod
    def _multipoint_gradient(x, no_points, axis = 0):
        """
        Compute a multipoint gradient over a specified axis of a matrix.

        Implemented using convolution with edges padded with zeros.

        Inputs:
            x -- 1D or 2D array
                Matrix or vector over which to compute the gradient.
            no_points -- even int >= 2
                Number of points to use to compute the gradient.
            axis -- int
                Axis over which to compute the gradient.

        Returns:
            Matrix with same shape as x with the first order multipoint gradient.
        """

        if no_points < 2 or no_points % 2 != 0:
            raise ValueError('no_points must be an even integer >= 2.')
        if x.ndim > 2:
            raise ValueError('not defined for x with more than 2 dimensions')

        kernel = -1 / np.concatenate(
            (np.arange(-(no_points - 1), 0, 2), np.arange(1, no_points + 1, 2))
        )
        kernel /= no_points // 2

        if x.ndim > 1:
            gradient = np.empty_like(x)
            slc = [slice(None)] * x.ndim
            for i in range(x.shape[1 - axis]):
                slc[1 - axis] = i
                gradient[slc] = np.convolve(x[slc].flatten(), kernel, 'same')
        else:
            gradient = np.convolve(x.flatten(), kernel, 'same')
            gradient = np.reshape(gradient, x.shape)

        return gradient

    def compute_gradient(self, no_points):
        """
        Compute the multipoint gradient of I for each sweep.
        Creates an attribute named `grad` containing the gradient.
        """
        self.grad = self._multipoint_gradient(self.I, no_points, axis = 0)

    @staticmethod
    def _find_peaks(x, height = None, threshold = None, distance = None,
        prominence = None, width = None, wlen = None, rel_height = 0.5, axis = 0):
        """Vectorized wrapper for scipy.signal.find_peaks.
        x must be a 1D or 2D array.
        Vectorized over axis.
        See scipy.signal.find_peaks docs for other args.
        """

        if x.ndim > 2:
            raise ValueError('x has {} dimensions, but a 1D or 2D array is required.'.format(x.ndim))

        elif x.ndim == 2:

            peaks = []
            properties = []

            slc = [slice(None)] * x.ndim
            for i in range(x.shape[1 - axis]):
                slc[1 - axis] = i
                peaks_tmp, props_tmp = signal.find_peaks(
                    x[slc], height, threshold, distance, prominence, width, wlen,
                    rel_height
                )
                peaks.append(peaks_tmp)
                properties.append(props_tmp)

        else:
            peaks, properties = signal.find_peaks(
                x.flatten(), height, threshold, distance, prominence, width, wlen,
                rel_height
            )

        return peaks, properties

    def find_grad_peaks(self, height_SD = 3.5, distance = 50):
        """Find peaks in the gradient.
        Creates attributes named `peak_threshold` (in mV/ms) and `peaks` (ms).
        Ensures that `peaks` is a list of arrays.
        """

        self.peak_threshold = self.grad.std() * height_SD
        peaks, _ = self._find_peaks(
            self.grad, height = self.peak_threshold, distance = distance
        )

        # Convert peaks to time from indices.
        try:
            # Check whether peaks is array-like.
            # Throws an AttributeError if peaks is not array-like (e.g. a list)
            assert peaks.ndim > 0
            peaks *= self.dt
            peaks = [peaks]
        except AttributeError:
            # If peaks is a list of arrays, multiply each array by dt.
            assert peaks[0].ndim > 0 # Check that the first entry is an array.
            peaks = [p * self.dt for p in peaks]

        # Check output type.
        assert type(peaks) is list
        assert peaks[0].ndim > 0

        self.peaks = peaks


    ### Support vectors & related methods
    @property
    def no_sweeps(self):
        return self.I.shape[1]

    @property
    def no_channels(self):
        return self.rec.shape[0]

    @property
    def t_vec(self):
        return np.arange(0, (self.I.shape[0] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self):
        return np.tile(self.t_vec[:, np.newaxis], (1, self.no_sweeps))

    @property
    def t_arr(self):
        return np.tile(self.t_mat[np.newaxis, :, :], (self.no_channels, 1, 1))


    ### Plotting methods

    def plot_rec(self):
        self.rec.plot(downsample = 1)

    def plot(self, sweeps = 'all', show_grad = True, show_grad_peaks = True):

        if sweeps == 'all':
            sweeps = slice(None)
            no_sweeps_plt = self.no_sweeps
        else:

            try:
                assert len(sweeps) == 2
                sweeps = slice(sweeps[0], sweeps[1])
                no_sweeps_plt = sweeps[1] - sweeps[0]
            except TypeError:
                # Catch case when sweeps is an int
                sweeps = sweeps
                no_sweeps_plt = 1

        plt.figure()

        ax = plt.subplot(111)
        plt.plot(self.t_mat[:, sweeps], self.I[:, sweeps], 'k-', alpha = min(1, 3/no_sweeps_plt))

        if show_grad:
            ax2 = ax.twinx()
            ax2.plot(self.t_mat[:, sweeps], self.grad[:, sweeps], 'r-', alpha = min(0.9, 5/no_sweeps_plt))
        if show_grad_peaks:
            if no_sweeps_plt > 1:
                tmp_grad = np.copy(self.grad[:, sweeps])
                for i, sw in enumerate(self.peaks[sweeps]):
                    ax2.plot(sw, tmp_grad[(sw / self.dt).round().astype(np.int32), i], 'go')

            else:
                ax2.plot(self.peaks[0], self.grad[(sw / self.dt).round().astype(np.int32), sweeps], 'go')



        plt.xlabel('Time (ms)')
        plt.ylabel('I')
        plt.show()

#%%

mini_detectors = []

for rec in mini_recs:
    tmp = MiniDetector(rec, remove_n_sweeps = 2)
    tmp.compute_gradient(50)
    tmp.find_grad_peaks()
    tmp.plot(0)
    mini_detectors.append(tmp)


#%% PRE-PROCESSING

minis = []

for rec in mini_recs:
    rec = rec[0, :, :] - rec[0, 4500:5000, :].mean(axis = 0)
    rec = rec[5000:, 1:]

    minis.append(rec)

    plt.figure()
    plt.plot(rec, 'k-', alpha = 0.7)
    plt.show()


#%% SANDBOX

test = minis[0]

plt.hist(test.flatten(), bins = 50)

plt.figure()
plt.hist(multipoint_gradient(test, 4, axis = 0).flatten(), bins = 50)
plt.show()

#%%

grad = multipoint_gradient(test, 200, axis = 0)

sw = 7

plt.figure()
ax = plt.subplot(111)
plt.plot(test[:, sw], 'k-')

ax2 = ax.twinx()
ax2.plot(grad[:, sw], 'r-', alpha = 0.7)
ax2.axhline(3.5 * grad.std(), color = 'k', ls = '--', dashes = (5, 5), lw = 0.5)

plt.show()
#%%

test_grad = multipoint_gradient(test, 200, axis = 0)
tmp = signal.argrelmax(test_grad, axis = 0, order = 200)

tmp[0]
tmp[1].max()
