#%% IMPORT MODULES

from __future__ import division

import copy
import pickle
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import seaborn as sns
from scipy import stats

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
sys.path.append('./src')
from FeedForwardDRN import SynapticKernel
from GIF import GIF
from AugmentedGIF import AugmentedGIF
from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from Filter_Exps import Filter_Exps
from Tools import timeToIntVec, generateOUprocess

#%% LOAD GIFS

GIFs = {}

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'rb') as f:
    GIFs['ser'] = pickle.load(f)
    f.close()

with open('./figs/scripts/gaba_neurons/opt_gaba_GIFs.pyc', 'rb') as f:
    GIFs['gaba'] = pickle.load(f)
    f.close()


#%% GET MEDIAN GIF PARAMETERS

mGIFs = {
    'ser': AugmentedGIF(0.1),
    'gaba': GIF(0.1)
}

mGIFs['ser'].Tref = 6.5
mGIFs['ser'].eta = Filter_Rect_LogSpaced()
mGIFs['ser'].eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
mGIFs['ser'].gamma = Filter_Exps()
mGIFs['ser'].gamma.setFilter_Timescales([30, 300, 3000])

mGIFs['gaba'].Tref = 4.0
mGIFs['gaba'].eta = Filter_Rect_LogSpaced()
mGIFs['gaba'].eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
mGIFs['gaba'].gamma = Filter_Exps()
mGIFs['gaba'].gamma.setFilter_Timescales([30, 300, 3000])

for celltype in mGIFs.keys():

    # Set coefficient values.
    for param in ['gl', 'El', 'C', 'gbar_K1', 'h_tau', 'gbar_K2', 'Vr', 'Vt_star', 'DV']:
        if getattr(GIFs[celltype][0], param, None) is None:
            print '{} mod does not have attribute {}. Skipping.'.format(celltype, param)
            continue

        tmp_param_ls = []
        for mod in GIFs[celltype]:
            tmp_param_ls.append(getattr(mod, param))
        setattr(mGIFs[celltype], param, np.median(tmp_param_ls))

    for kernel in ['eta', 'gamma']:
        tmp_param_arr = []
        for mod in GIFs[celltype]:
            tmp_param_arr.append(getattr(mod, kernel).getCoefficients())
        vars(mGIFs[celltype])[kernel].setFilter_Coefficients(np.median(tmp_param_arr, axis = 0))

#%%

def plot_sample(sample):
    spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.1])

    plt.figure(figsize = (5, 4))

    plt.subplot(spec[0, :])
    plt.plot(sample['t'], sample['I'], '-', color = 'gray')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.xticks([])
    plt.ylabel('$I$ (nA)')

    plt.subplot(spec[1, :])
    plt.plot(sample['t'], sample['V'], 'k-')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.xticks([])
    plt.ylabel('$V$ (mV)')

    plt.subplot(spec[2, :])
    plt.plot(sample['spks'], np.zeros_like(sample['spks']), 'k|')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.ylabel('Repeat no.')

    plt.tight_layout()


mGIFs['ser'].gbar_K1 = 0.#010
mGIFs['ser'].h_tau = 50.
mGIFs['ser'].gbar_K2 = 0.002
mGIFs['ser'].DV = 2.5

### Simulation parameters
dt = 0.1
distal_in = SynapticKernel('alpha', tau = 15, ampli = 1, kernel_len = 500, dt = dt).centered_kernel
gaba_kernel = SynapticKernel('alpha', tau = 25, ampli = -0.005, kernel_len = 400, dt = dt).centered_kernel

no_ser_neurons = 600
no_gaba_neurons = 400
connectivity_prob = 5. / no_gaba_neurons
prop_delay_int = int(2 / dt)
scales = np.linspace(0.02, 0.20, 18)

### Create structure to hold output
ff_results = pd.DataFrame(columns = ['Scale', 'Inhibition', 'Input', 'ser spks', 'ser ex', 'gaba ex'])

### Initialize random components
OU_noise = {
    'ser': np.array(
        [generateOUprocess(len(distal_in) * dt, 3, 0, 0.020, dt, random_seed = int(i + 1)) for i in range(no_ser_neurons)]
    ),
    'gaba': np.array(
        [generateOUprocess(len(distal_in) * dt, 3, 0, 0.020, dt, random_seed = int((i + 1) / np.pi)) for i in range(no_gaba_neurons)]
    )
}

connectivity_matrix = (
    np.random.uniform(size = (no_ser_neurons, no_gaba_neurons)) < connectivity_prob
).astype(np.int8)

### Perform simulation
cnt = 0
for scale in scales:

    I = scale * distal_in

    # Iterate over GABA neurons providing input to a single 5HT cell.
    gaba_outmat = np.empty((no_gaba_neurons, len(distal_in)), dtype = np.float32)
    for gaba_no in range(no_gaba_neurons):
        t, V, eta, v_T, spks = mGIFs['gaba'].simulate(
            I + OU_noise['gaba'][gaba_no, :], mGIFs['gaba'].El
        )
        gaba_outmat[gaba_no, :] = np.convolve(
            timeToIntVec(spks, distal_in.shape[0] * dt, dt), gaba_kernel, 'same'
        ).astype(np.float32)

        # Save a sample trace.
        if gaba_no == 0:
            gaba_ex = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

        print '\rSimulating GABA neurons for scale {:.4f} - {:.1f}%'.format(scale, 100 * (gaba_no + 1) / no_gaba_neurons),
    print '\n',

    # Add propagation delay to GABA input.
    gaba_outmat = np.roll(gaba_outmat, prop_delay_int, axis = 1)
    gaba_outmat[:, :prop_delay_int] = 0

    # Transform GABA output into 5HT input using connectivity matrix.
    gaba_inmat = np.dot(connectivity_matrix, gaba_outmat)

    # Allocate arrays for 5HT spiketrains.
    ser_spkvecs = {
        'ib': np.zeros_like(distal_in, dtype = np.int16),
        'reg': np.zeros_like(distal_in, dtype = np.int16)
    }
    # Create a dict to hold 5HT example traces.
    ser_examples = {}

    for ser_no in range(no_ser_neurons):
        #I_tmp = I + OU_noise['ser'][ser_no, :]
        #print hex(id(I_tmp))

        # Try with and without feed-forward inhibition
        for ib_status, ib_multiplier in zip(('ib', 'reg'), (1, 0)):
            t, V, eta, v_T, spks = mGIFs['ser'].simulate(
                I + OU_noise['ser'][ser_no, :] + gaba_inmat[ser_no, :] * ib_multiplier,
                mGIFs['ser'].El
            )
            ser_spkvecs[ib_status] += timeToIntVec(spks, distal_in.shape[0] * dt, dt)

            # Save a sample trace.
            if ser_no == 0:
                ser_examples[ib_status] = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

        print '\rSimulating 5HT neurons for scale {:.4f} - {:.1f}%'.format(scale, 100 * (ser_no + 1) / no_ser_neurons),
    print '\n'


    for ib_status in ('ib', 'reg'):
        row = {
            'Scale': scale,
            'Inhibition': ib_status,
            'Input': I,
            'ser spks': ser_spkvecs[ib_status],
            'ser ex': ser_examples[ib_status],
            'gaba ex': gaba_ex
        }
        ff_results = ff_results.append(row, ignore_index = True)

    #gc.collect()
    cnt += 1

#%%

if True:
    def PSTH(spktrain, window_width, no_neurons, dt = 0.1):
        """
        Obtain the population firing rate with a resolution of `window_width`.
        """
        kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
        psth = np.convolve(spktrain, kernel, 'same')
        return psth

    IMG_PATH = './figs/ims/ff_drn/'
    plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

    plt.figure(figsize = (6, 3))

    axes = {
        'reg': plt.subplot(131),
        'ib': plt.subplot(132),
        'fi': plt.subplot(133)
    }

    fi_tmp = []

    for i in range(ff_results.shape[0]):

        psth_tmp = PSTH(ff_results.loc[i, 'ser spks'], 0.010, no_ser_neurons, 0.0001)
        axes[ff_results.loc[i, 'Inhibition']].plot(
            psth_tmp,
            'k-', lw = 0.5
        )

        fi_tmp.append(psth_tmp.max())

    ff_results['PSTH peak'] = fi_tmp
    del fi_tmp

    axes['fi'].plot(
        ff_results.loc[ff_results['Inhibition'] == 'reg', 'Scale'],
        ff_results.loc[ff_results['Inhibition'] == 'reg', 'PSTH peak'],
        'k-', label = 'Ctrl'
    )
    axes['fi'].plot(
        ff_results.loc[ff_results['Inhibition'] == 'ib', 'Scale'],
        ff_results.loc[ff_results['Inhibition'] == 'ib', 'PSTH peak'],
        'r-', label = 'Feed-forw. ib.'
    )
    axes['fi'].legend(loc = 'upper left')
    axes['fi'].set_ylabel('PSTH peak')
    axes['fi'].set_xlabel('Input amplitude (nA)')
    axes['fi'].set_ylim(0, 90)

    axes['reg'].set_title('No inhibition')
    axes['reg'].set_ylabel('Pop. firing rate (Hz)')
    axes['reg'].set_xlabel('Time (timesteps)')
    axes['reg'].set_xlim(5000, 6000)
    axes['reg'].set_ylim(0, 90)
    axes['ib'].set_title('Feed forward inhibition')
    axes['ib'].set_ylabel('Pop. firing rate (Hz)')
    axes['ib'].set_xlabel('Time (timesteps)')
    axes['ib'].set_xlim(5000, 6000)
    axes['ib'].set_ylim(0, 90)

    plt.tight_layout()

    if True and IMG_PATH is not None:
        plt.savefig(IMG_PATH + 'prelim_ff_15_bg_noise_noIA.png')

    plt.show()


#%%


if True:
    plot_sample(ff_results.loc[
        np.logical_and(
            np.abs(ff_results['Scale'] - 0.080) < 1e-3,
            ff_results['Inhibition'] == 'ib'
        ), 'ser ex'
    ].tolist()[0])

    plt.savefig(IMG_PATH + 'ser_example_noise_45.png')

#%%

plt.figure(figsize = (5, 3))
plt.subplot(121)
plt.title('Connectivity matrix')
plt.imshow(connectivity_matrix)
plt.ylabel('5HT neuron number')
plt.xlabel('GABA neuron number')

plt.subplot(122)
plt.title('No. GABA inputs/5HT cell')
plt.hist(connectivity_matrix.sum(axis = 1), density = True)
plt.ylabel('Density')
plt.xlabel('No. GABA inputs')

plt.tight_layout()

plt.savefig(IMG_PATH + 'sample_conn_mat.png')
plt.show()

#%%

"""
peak amplitude, time to peak, mean time, sd time, skew time, kurtosis time

input starts at 500.0 ms
"""

input_start_time = 500.0

(ff_results.loc[i, 'ser spks'] > 2).sum()
ff_results.loc[0, 'Inhibition']

spk_latencies = {
    'ib': [],
    'reg': []
}

for i in range(ff_results.shape[0]):
    latencies_tmp = []
    for ind in np.where(ff_results.loc[i, 'ser spks'] > 0)[0]:
        if ind < int(input_start_time/dt):
            continue
        else:
            latencies_tmp.extend(
                [ind * 0.1 - input_start_time] * ff_results.loc[i, 'ser spks'][ind]
            )

    spk_latencies[ff_results.loc[i, 'Inhibition']].append(latencies_tmp)


#%%

plt.figure(figsize = (6, 3))

plt.subplot(131)
plt.title('Location')
plt.semilogy(
    scales, [np.mean(x) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [np.mean(x) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
plt.ylabel('Mean latency (ms)')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.subplot(132)
plt.title('Variability')
plt.semilogy(
    scales, [np.std(x) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [np.std(x) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
plt.ylabel('Latency SD (ms)')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.subplot(133)
plt.title('Asymmetry')
plt.semilogy(
    scales, [stats.skew(x) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [stats.skew(x) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
plt.ylabel('Latency skewness (ms)')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'latency_shape_param_15_nIA.png')

plt.show()


#%%

plt.figure(figsize = (6, 3))

plt.subplot(131)
plt.title('Location')
plt.semilogy(
    scales, [np.median(x) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [np.median(x) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
#plt.ylim(0, plt.ylim()[1])
plt.ylabel('Median latency (ms)')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.subplot(132)
plt.title('Variability')
plt.semilogy(
    scales, [np.percentile(x, 75) - np.percentile(x, 25) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [np.percentile(x, 75) - np.percentile(x, 25) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
#plt.ylim(0, plt.ylim()[1])
plt.ylabel('Interquartile range (ms)')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.subplot(133)
plt.title('Asymmetry')
plt.semilogy(
    scales, [(
        np.percentile(x, 75) - np.percentile(x, 50)
        / np.percentile(x, 50) - np.percentile(x, 25)
    ) for x in spk_latencies['reg']],
    'k-', label = 'Control'
)
plt.semilogy(
    scales, [(
        np.percentile(x, 75) - np.percentile(x, 50)
        / np.percentile(x, 50) - np.percentile(x, 25)
    ) for x in spk_latencies['ib']],
    'r-', label = 'Feed-forward ib.'
)
plt.axhline(1, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
#plt.ylim(0, plt.ylim()[1])
plt.ylabel(r'Latency $\frac{Q_3 - Q_2}{Q_2 - Q_1}$')
plt.xlabel('Input amplitude (nA)')
plt.legend()

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'latency_shape_nonparam_15_nIA.png')

plt.show()
