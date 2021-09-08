from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from scipy import signal

from .Tools import timeToIndex


class Trace:

    """
    An experiments includes many experimental traces.
    A Trace contains the experimental data acquired during a single current-clamp injection (e.g., the training set injection, or one repetition of the test set injections)
    """

    def __init__(self, V, I, T, dt):
        """
        V : vector with recorded voltage (mV)
        I : vector with injected current (nA)
        T : length of the recording (ms)
        dt : timestep of recording (ms)
        """

        # Perform input checks
        if len(V) != len(I):
            raise ValueError(
                'Could not create Trace using V and I with non-'
                'identical lengths {} and {}.'.format(len(V), len(I))
            )
        if len(V) != int(np.round(T / dt)):
            warn(
                RuntimeWarning(
                    'V array is not of length T/dt; expected {}, '
                    'got {}.'.format(int(np.round(T / dt)), len(V))
                )
            )

        # Initialize main attributes related to recording
        self.V_rec = np.array(
            V, dtype='double'
        )  # mV, recorded voltage (before AEC)
        self.V = self.V_rec  # mV, voltage (after AEC)
        self.I = np.array(I, dtype='double')  # nA, injected current
        self.T = T  # ms, duration of the recording
        self.dt = dt  # ms, timestep

        # Initialize flags
        self.AEC_flag = False  # Has the trace been preprocessed with AEC?

        self.filter_flag = False

        self.spks_flag = False  # Do spikes have been detected?
        self.spks = 0  # spike indices stored in indices (and not in ms!)

        self.useTrace = True  # if false this trace will be neglected while fitting spiking models to data
        self.ROI = [
            [0, len(self.V_rec) * self.dt]
        ]  # List of intervals to be used for fitting; includes the whole trace by default

    #################################################################################################
    # FUNCTIONS ASSOCIATED WITH FILTERING
    #################################################################################################

    def butterLowpassFilter(self, cutoff, order=3):
        """
        Apply a Butterworth lowpass filter to V_rec and I.

        Inputs:
            cutoff      --  (Hz) critical frequency of the filter
            order       --  filter order


        Modifies Trace.V_rec and Trace.I inplace. Raises an error if signals have already been filtered.

        Note: because this method affects Trace.V_rec and not Trace.V it must be called before performing AEC!
        """

        if self.filter_flag:
            raise RuntimeError('Trace already filtered!')

        # Convert frequencies to radians.
        cutoff *= 2.0 * np.pi
        sampling_rate = 2.0 * np.pi / (self.dt / 1000.0)

        # Define cutoff frequency.
        nyq = 0.5 * sampling_rate
        normal_cutoff = cutoff / nyq

        # Get filter coefficients.
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Filter data.
        self.V_rec[100:] = signal.lfilter(b, a, self.V_rec)[100:]
        self.I[100:] = signal.lfilter(b, a, self.I)[100:]

        self.filter_flag = True

    #################################################################################################
    # FUNCTIONS ASSOCIATED WITH ROI
    #################################################################################################

    def enable(self):
        """
        If you want to use this trace call this function. By default traces are enabled.
        When enable is called, ROI is set to be the entire trace.
        """

        self.useTrace = True
        self.ROI = [[0, len(self.V_rec) * self.dt]]

    def disable(self):
        """
        If you dont want to use this trace during the fit, call this function.
        """

        self.useTrace = False
        self.ROI = [[0, 0]]

    def setROI(self, ROI_intervals):
        """
        ROI intervals are defined in ms.
        Use this function the specify which parts of the trace have to be used for fitting.
        """
        self.useTrace = True
        self.ROI = ROI_intervals

    def setROI_Bool(self, vector):

        prev_ROI_vec = np.zeros_like(self.V, dtype=np.bool)
        prev_ROI_vec[self.getROI()] = True

        assert (
            prev_ROI_vec.shape == vector.shape
        ), 'ROI vectors do not have same shape.'

        new_ROI_vec = np.logical_and(prev_ROI_vec, vector)

        rising_edges, falling_edges = self._getBoolEdges(new_ROI_vec, self.dt)

        ROI_intervals = []

        if rising_edges[0] > falling_edges[0]:
            rising_edges = np.concatenate(([0.0], rising_edges))

        for i in range(min((len(rising_edges), len(falling_edges)))):
            ROI_intervals.append([rising_edges[i], falling_edges[i]])

        self.ROI = ROI_intervals

    @staticmethod
    @nb.jit
    def _getBoolEdges(vector, dt):
        """
        Get timestamps of rising/falling edges of boolean vector.
        """

        rising_edges = np.zeros(len(vector) / 2, dtype=np.float64)
        re_cnt = 0

        falling_edges = np.zeros(len(vector) / 2, dtype=np.float64)
        fe_cnt = 0

        for t_step in range(1, len(vector)):

            if vector[t_step] != vector[t_step - 1]:
                # Check whether edge is rising or falling only if
                # there is an edge.

                if vector[t_step]:
                    # Rising edge.
                    rising_edges[re_cnt] = float(t_step) * dt
                    re_cnt += 1

                else:
                    # Falling edge.
                    falling_edges[fe_cnt] = float(t_step) * dt
                    fe_cnt += 1

        return (rising_edges[:re_cnt], falling_edges[:fe_cnt])

    def getROI(self):
        """
        Return indices of the trace which are in ROI
        """

        ROI_region = np.zeros(int(self.T / self.dt), dtype=np.bool)

        for ROI_interval in self.ROI:
            ROI_region[
                int(ROI_interval[0] / self.dt) : int(ROI_interval[1] / self.dt)
            ] = True

        ROI_ind = np.where(ROI_region)[0]

        # Make sure indices are ok
        ROI_ind = ROI_ind[np.where(ROI_ind < int(self.T / self.dt))[0]]

        return ROI_ind

    def getROI_FarFromSpikes(self, DT_before, DT_after):
        """
        Return indices of the trace which are in ROI. Exclude all datapoints which are close to a spike.
        DT_before: ms
        DT_after: ms
        These two parameters define the region to cut around each spike.
        """

        L = len(self.V)

        LR_flag = np.ones(L, dtype=np.bool)

        # Select region in ROI
        ROI_ind = self.getROI()
        LR_flag[ROI_ind] = False

        # Remove spks-associated indices iff there are spks in the Trace
        if len(self.spks) >= 1:

            # Remove region around spikes
            DT_before_i = int(DT_before / self.dt)
            DT_after_i = int(DT_after / self.dt)

            for s in self.spks:

                lb = max(0, s - DT_before_i)
                ub = min(L, s + DT_after_i)

                LR_flag[lb:ub] = True

        indices = np.where(~LR_flag)[0]

        return indices

    def getROI_cutInitialSegments(self, DT_initialcutoff):
        """
        Return indices of the trace which are in ROI. Exclude all initial segments in each ROI.
        DT_initialcutoff: ms, width of region to cut at the beginning of each ROI section.
        """

        DT_initialcutoff_i = int(DT_initialcutoff / self.dt)
        ROI_region = np.zeros(int(self.T / self.dt), dtype=np.bool)

        for ROI_interval in self.ROI:

            lb = int(ROI_interval[0] / self.dt) + DT_initialcutoff_i
            ub = int(ROI_interval[1] / self.dt)

            if lb < ub:
                ROI_region[lb:ub] = True

        ROI_ind = np.where(ROI_region)[0]

        # Make sure indices are ok
        ROI_ind = ROI_ind[np.where(ROI_ind < int(self.T / self.dt))[0]]

        return ROI_ind

    #################################################################################################
    # FUNCTIONS ASSOCIATED TO SPIKES IN THE TRACE
    #################################################################################################

    def detectSpikes(self, threshold=0.0, ref=3.0):
        """
        Detect action potentials by threshold crossing (parameter threshold, mV) from below (i.e. with dV/dt>0).
        To avoid multiple detection of same spike due to noise, use an 'absolute refractory period' ref, in ms.
        """
        ref_ind = int(np.round(ref / self.dt))
        spk_inds = getRisingEdges(self.V, threshold, ref_ind)

        self.spks = spk_inds
        self.spks_flag = True

        self.spks_flag = True

    def computeAverageSpikeShape(self):
        """
        Compute the average spike shape using spikes in ROI.
        """

        DT_before = 10.0
        DT_after = 20.0

        DT_before_i = int(DT_before / self.dt)
        DT_after_i = int(DT_after / self.dt)

        if not self.spks_flag:
            self.detectSpikes()

        all_spikes = []

        ROI_ind = self.getROI()

        for s in self.spks:

            # Only spikes in ROI are used
            if s in ROI_ind:

                # Avoid using spikes close to boundaries to avoid errors
                if s > DT_before_i and s < (len(self.V) - DT_after_i):
                    all_spikes.append(self.V[s - DT_before_i : s + DT_after_i])

        spike_avg = np.mean(all_spikes, axis=0)

        support = np.linspace(-DT_before, DT_after, len(spike_avg))
        spike_nb = len(all_spikes)

        return (support, spike_avg, spike_nb)

    def getSpikeTrain(self):
        """
        Return spike train defined as a vector of 0s and 1s. Each bin represent self.dt
        """

        spike_train = np.zeros(int(self.T / self.dt))

        if len(self.spks) > 0:

            spike_train[self.spks] = 1

        return spike_train

    def getSpikeTimes(self):
        """
        Return spike times in units of ms.
        """

        return self.spks * self.dt

    def getSpikeTimesInROI(self):
        return self.getSpikeIndicesInROI() * self.dt

    def getSpikeIndices(self):
        """
        Return spike indices in units of dt.
        """

        return self.spks

    def getSpikeIndicesInROI(self):
        return np.intersect1d(self.getSpikeIndices(), self.getROI(), True)

    def getSpikeNb(self):

        return len(self.spks)

    def getSpikeNbInROI(self):
        """
        Return number of spikes in region of interest.
        """

        ROI_ind = self.getROI()
        spike_train = self.getSpikeTrain()

        return sum(spike_train[ROI_ind])

    #################################################################################################
    # POWER SPECTRUM ANALYSIS
    #################################################################################################

    def extractPowerSpectrumDensity(self, do_plot=False):
        """
        Estimate the power spectrum density of the recorded voltage

        Returns tuple of vectors: frequency (Hz) and corresponding power spectrum density.
        """

        f_V, PSD_V = signal.welch(
            self.V, 1000.0 / self.dt, window='hanning', nperseg=100000
        )

        f_I, PSD_I = signal.welch(
            self.I, 1000.0 / self.dt, window='hanning', nperseg=100000
        )

        if do_plot:

            plt.figure(figsize=(8, 6))

            ax = plt.subplot(211)
            ax.set_xscale('log')

            ax.plot(f_V, PSD_V, 'k-', linewidth=0.5, label='V')
            ax.plot(f_I, PSD_I, 'b-', linewidth=0.5, label='I')

            ax.legend()
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')

            ax2 = plt.subplot(212, sharex=ax)
            ax2.set_xscale('log')

            ax2.plot(
                f_V, PSD_V / PSD_I, 'k-', linewidth=0.5, label='V (norm.)'
            )

            ax2.legend()
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('PSD')

            plt.tight_layout()
            plt.show()

        return f_V, PSD_V, f_I, PSD_I

    #################################################################################################
    # GET STATISTICS, COMPUTED IN ROI
    #################################################################################################

    def getSpikeNb_inROI(self):

        if len(self.spks) == 0:

            return 0

        else:

            spike_train = self.getSpikeTrain()
            ROI_ind = self.getROI()

            nbSpikes = np.sum(spike_train[ROI_ind])

            return nbSpikes

    def getTraceLength_inROI(self):
        """
        Return in ms the duration of ROI region.
        """
        ROI_ind = self.getROI()

        return len(ROI_ind) * self.dt

    def getFiringRate_inROI(self):
        """
        Return the average firing rate (in Hz) in ROI.
        """

        return 1000.0 * self.getSpikeNb_inROI() / self.getTraceLength_inROI()

    #################################################################################################
    # GET TIME
    #################################################################################################

    def getTime(self):
        """
        Get time vector (i.e., the temporal support of the arrays: I, V, etc)
        """

        return np.arange(int(self.T / self.dt)) * self.dt

    #################################################################################################
    # FUNCTIONS ASSOCIATED WITH PLOTTING
    #################################################################################################

    def plot(self):
        """
        Plot input current, recorded voltage, voltage after AEC (is applicable) and detected spike times (if applicable)
        """

        time = self.getTime()

        plt.figure(figsize=(10, 4), facecolor='white')

        plt.subplot(2, 1, 1)
        plt.plot(time, self.I, 'gray')
        plt.ylabel('I (nA)')

        plt.subplot(2, 1, 2)
        plt.plot(time, self.V_rec, 'black')

        if self.AEC_flag:
            plt.plot(time, self.V, 'red')

        if self.spks_flag:
            plt.plot(
                self.getSpikeTimes(),
                np.zeros(len(self.spks)),
                '.',
                color='blue',
            )

        # Plot ROI (region selected for performing operations)
        ROI_vector = 100.0 * np.ones(len(self.V))
        ROI_vector[self.getROI()] = -100.0
        plt.fill_between(self.getTime(), ROI_vector, -100.0, color='0.2')

        plt.ylim([min(self.V) - 5.0, max(self.V) + 5.0])
        plt.ylabel('V rec (mV)')
        plt.xlabel('Time (ms)')
        plt.show()


def filterTimesByROI(times, ROI):
    """Get subset of timestamps that are within ROI.

    Arguments
    ---------
    times : list-like of floats (or list of lists-like of floats)
        Timestamps (ms) to filter.
    ROI : pair of floats
        Start and end (ms) of inclusion window for timestamps.

    """
    try:
        # Case that times is a list of lists.
        iter(times[0])  # Check that first entry in times is list-like.
        filteredTimes = []
        for sweep in times:
            filteredTimes.append(_unvectorizedFilterTimesByROI(sweep, ROI))
    except TypeError:
        # Case that times is a list of spike times.
        filteredTimes = _unvectorizedFilterTimesByROI(times, ROI)

    return filteredTimes


def _unvectorizedFilterTimesByROI(times, ROI):
    assert np.ndim(times) == 1
    assert len(ROI) == 2
    filteredTimes = np.asarray(
        [time for time in times if time >= ROI[0] and time < ROI[1]]
    )
    return filteredTimes


def getRisingEdges(x, threshold, debounce):
    """Get indices where x has risen across a threshold.

    Arguments
    ---------
    x : 1d numeric array
    threshold : float
    debounce : int
        Ignore threshold crossings for debounce indices after each detected
        rising edge. Avoids the same rising edge being detected several times
        due to signal noise.

    Returns
    -------
    1d int array of indices at which x has risen across threshold.

    """
    # Get indices of edges rising across threshold.
    above_thresh = np.asarray(x) >= threshold
    below_thresh = ~above_thresh
    rising_edges_bool = above_thresh[1:] & below_thresh[:-1]
    rising_edges_inds = np.where(rising_edges_bool)[0] + 1

    # Debounce.
    if len(rising_edges_inds) >= 1:
        redundant_pts = np.where(np.diff(rising_edges_inds) <= debounce)[0] + 1
        rising_edges_inds_debounced = np.delete(
            rising_edges_inds, redundant_pts
        )
    else:
        rising_edges_inds_debounced = rising_edges_inds

    return rising_edges_inds_debounced


def detectSpikes(arr, threshold, ref, axis, dt=0.1):
    """Detect spikes in a voltage array.

    Vectorized implementation.

    Arguments
    ---------
    arr : float vector or matrix-like
        Array of voltages in which to detect spikes.
    threshold : float
        Voltage threshold for spike detection.
    ref : float
        Absolute refractory period (ms). Avoids the same spike being detected
        multiple times due to recording noise.
    axis : -1, 0, or 1
        Time axis of array. -1 flattens the array before detection.
    dt : float, default 0.1
        Timestep of recording (ms).

    Returns
    -------
    Nested list of spike times (ms). If axis=-1, a flat list is returned.

    """
    # Input checks.
    if np.ndim(arr) > 2:
        raiseExpectedGot(
            'vector or matrix-like',
            'argument `arr`',
            '{}d array'.format(np.ndim(arr)),
        )
    if not any([axis == valid_axis for valid_axis in (-1, 0, 1)]):
        raiseExpectedGot('one of [-1, 0, 1]', 'argument `axis`', axis)

    # Coerce arr to numpy array and prep for vectorization over rows.
    if axis == -1:
        arr = np.array(arr, copy=True).flatten()[np.newaxis, :]
    elif axis == 0:
        arr = np.asarray(arr).T
    elif axis == 1:
        arr = np.asarray(arr)
    else:
        # Cannot get here.
        raise RuntimeError('Unexpectedly reached end of switch.')

    # Get spike times.
    spktimes = []
    for i in range(arr.shape[0]):
        row_spktimes = (
            getRisingEdges(arr[i, :], threshold, timeToIndex(ref, dt)[0]) * dt
        )
        spktimes.append(row_spktimes)

    # Flatten result if axis == -1
    if axis == -1:
        spktimes = np.asarray(spktimes).flatten()

    return spktimes
