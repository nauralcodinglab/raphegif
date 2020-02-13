import numpy as np

from .Trace import getRisingEdges

def getIndicesFarFromSpikes(T, spikes_i, dt_before, dt_after, initial_cutoff, dt):

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_before_i = int(dt_before/dt)
    dt_after_i = int(dt_after/dt)

    for s in spikes_i:
        flag[max(s-dt_before_i, 0): min(s+dt_after_i, T_i)] = 1

    selection = np.where(flag == 0)[0]

    return selection


def getIndicesDuringSpikes(T, spikes_i, dt_after, initial_cutoff, dt):

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_after_i = int(dt_after/dt)

    for s in spikes_i:
        flag[max(s, 0): min(s+dt_after_i, T_i)] = 1

    selection = np.where(flag > 0.1)[0]

    return selection


def PSTH(spktrain, window_width, no_neurons, dt=0.1):
    """
    Obtain the population firing rate with a resolution of `window_width`.
    """

    window_width *= 1e-3
    dt *= 1e-3

    kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
    psth = np.convolve(spktrain, kernel, 'same')
    return psth


def getSpikeLatency(voltage, start_time, threshold=0., refractory_period=3., dt=0.1):
    """Get the time to the first spike after start_time.

    Returns NaN if there are no spikes after start_time, or if there are any
    spikes before start_time.

    Arguments
    ---------
    voltage : 1d float array-like
    start_time : float
    threshold : float, default 0.
        Voltage threshold for spike detection.
    refractory_period: float, default 3.
        Absolute refractory period for spike detection. Avoids detecting the
        same spike multiple times due to noise.
    dt : float, default 0.1
        Timestep of recording (ms).

    Returns
    -------
    float time from start_time until first spike (if any).

    """
    spike_inds = getRisingEdges(voltage, threshold, refractory_period)
    spike_times = spike_inds * dt

    if np.any(spike_times <= start_time):
        latency = np.nan
    elif not np.any(spike_times > start_time):
        latency = np.nan
    else:
        latency = np.min(spike_times - start_time)

    return latency
