import numpy as np
import weave

from .Trace import getRisingEdges


def getIndicesFarFromSpikes(
    T, spikes_i, dt_before, dt_after, initial_cutoff, dt
):

    T_i = int(T / dt)
    flag = np.zeros(T_i)
    flag[: int(initial_cutoff / dt)] = 1
    flag[-1] = 1

    dt_before_i = int(dt_before / dt)
    dt_after_i = int(dt_after / dt)

    for s in spikes_i:
        flag[max(s - dt_before_i, 0) : min(s + dt_after_i, T_i)] = 1

    selection = np.where(flag == 0)[0]

    return selection


def getIndicesDuringSpikes(T, spikes_i, dt_after, initial_cutoff, dt):

    T_i = int(T / dt)
    flag = np.zeros(T_i)
    flag[: int(initial_cutoff / dt)] = 1
    flag[-1] = 1

    dt_after_i = int(dt_after / dt)

    for s in spikes_i:
        flag[max(s, 0) : min(s + dt_after_i, T_i)] = 1

    selection = np.where(flag > 0.1)[0]

    return selection


def PSTH(spktrain, window_width, no_neurons, dt=0.1):
    """Obtain the population firing rate with a resolution of `window_width`.

    Uses either np.convolve or an accelerated sparse convolution method
    depending on the sparsity of spktrain.

    """
    # Based on rough benchmarks, the sparse method is faster when the
    # spktrain is >65% zeros.
    if (np.asarray(spktrain) == 0).sum() > (0.65 * len(spktrain)):
        # Use sparse method if spktrain is more than 65% zeros.
        return _sparse_PSTH(spktrain, window_width, no_neurons, dt=dt)
    else:
        # Use dense method if spktrain is <= 65% zeros.
        return _dense_PSTH(spktrain, window_width, no_neurons, dt=dt)


def _dense_PSTH(spktrain, window_width, no_neurons, dt=0.1):
    """Obtain the population firing rate with a resolution of 
    `window_width` using np.convolve.

    Runtime is not affected by sparsity of spktrain.

    """

    window_width *= 1e-3
    dt *= 1e-3

    kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
    psth = np.convolve(spktrain, kernel, 'same')
    return psth


def _sparse_PSTH(spktrain, window_width, no_neurons, dt=0.1):
    """Obtain the population firing rate with a resolution of `window_width`
    using a sparse convolution method.

    Runtime decreases with increasing sparsity of spktrain. Up to ~30X
    faster than dense method for very sparse spktrains.

    """
    window_width *= 1e-3
    dt *= 1e-3

    p_kernel_length = int(window_width / dt)
    p_kernel_weight = 1.0 / (window_width * no_neurons)
    psth = np.zeros(len(spktrain) + p_kernel_length, dtype=np.float32)
    spktrain = np.asarray(spktrain).astype(np.float32)
    p_num_spktrain_timesteps = len(spktrain)

    code = """
    float kernel_weight = float(p_kernel_weight);
    int kernel_length = int(p_kernel_length);
    int num_spktrain_timesteps = int(p_num_spktrain_timesteps);

    float num_spikes;  // Number of spikes in a single timestep.
    for (int i=0; i<int(num_spktrain_timesteps); i++) {
        num_spikes = spktrain[i];
        if (num_spikes > 0.0) {

            // Add kernel to PSTH.
            float firing_rate_to_add = num_spikes * kernel_weight;
            for (int j=0; j<kernel_length; j++) {
                psth[i+j] += firing_rate_to_add;
            }
        }
    }
    """

    vars = [
        'p_kernel_length',
        'p_kernel_weight',
        'psth',
        'spktrain',
        'p_num_spktrain_timesteps',
    ]

    v = weave.inline(code, vars)

    # Select part of PSTH to return that will match results
    # of np.convolve(mode='same'). Very weird, but necessary
    # to get same results.
    half_kernel_timesteps = p_kernel_length // 2
    other_half_kernel_timesteps = p_kernel_length - half_kernel_timesteps
    return psth[
        max(0, other_half_kernel_timesteps - 1) : (
            len(psth) - half_kernel_timesteps - 1
        )
    ]


def getSpikeLatency(
    voltage, start_time, threshold=0.0, refractory_period=3.0, dt=0.1
):
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
