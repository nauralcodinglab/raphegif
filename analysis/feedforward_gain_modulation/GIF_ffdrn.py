#%% IMPORT MODULES

from __future__ import division

import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

import sys
sys.path.append('./analysis/feedforward_gain_modulation')
sys.path.append('./src')
from FeedForwardDRN import SynapticKernel
from GIF import GIF
from AugmentedGIF import AugmentedGIF
from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from Filter_Exps import Filter_Exps
from Tools import timeToIntVec

#%% LOAD GIFS

GIFs = {}

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'rb') as f:
    GIFs['ser'] = pickle.load(f)
    f.close()

with open('./figs/scripts/gaba_neurons/opt_gaba_GIFs.pyc', 'rb') as f:
    GIFs['gaba'] = pickle.load(f)
    f.close()


#%% GET MEAN GIF PARAMETERS

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

#%%

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

    mGIFs[celltype].printParameters()
    mGIFs[celltype].plotParameters()

#%% SIMPLE CURRENT STEP SIMULATION

IMG_PATH = './figs/ims/ff_drn/'
plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

mGIFs['ser'].gbar_K2 = 0.002

I = np.concatenate((np.zeros(1500), -0.035 * np.ones(3500), 0.1 * np.ones(10000)))

t, V, eta, v_T, spks = mGIFs['ser'].simulate(I, mGIFs['ser'].El)

spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.6])

plt.figure()

plt.subplot(spec[0, :])
plt.plot(t, I, '-', color = 'gray')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec[1, :])
plt.plot(t, V, 'k-')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$V$ (mV)')

plt.subplot(spec[2, :])
plt.plot(spks, np.zeros_like(spks), 'k|')
for i in range(20):
    t, _, _, _, spks = mGIFs['ser'].simulate(I, mGIFs['ser'].El)
    plt.plot(spks, (i + 1) * np.ones_like(spks), 'k|')
plt.xlim(t[0], t[-1])
plt.ylabel('Repeat no.')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'median_KGIF_gk2_0.002.png')

plt.show()


#%%

t, V, eta, v_T, spks = mGIFs['gaba'].simulate(I, mGIFs['gaba'].El)

spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.6])

plt.figure()

plt.subplot(spec[0, :])
plt.plot(t, I, '-', color = 'gray')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec[1, :])
plt.plot(t, V, 'k-')
plt.xlim(t[0], t[-1])
plt.xticks([])
plt.ylabel('$V$ (mV)')

plt.subplot(spec[2, :])
plt.plot(spks, np.zeros_like(spks), 'k|')
for i in range(20):
    t, _, _, _, spks = mGIFs['gaba'].simulate(I, mGIFs['gaba'].El)
    plt.plot(spks, (i + 1) * np.ones_like(spks), 'k|')
plt.xlim(t[0], t[-1])
plt.ylabel('Repeat no.')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'median_gabaGIF.png')

plt.show()

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

    plt.show()

mGIFs['ser'].gbar_K2 = 0.002

### Simulation parameters
distal_in = SynapticKernel('alpha', tau = 45, ampli = 1, kernel_len = 500, dt = 0.1).centered_kernel
gaba_kernel = SynapticKernel('alpha', tau = 25, ampli = -0.005, kernel_len = 400, dt = 0.1).centered_kernel

no_reps = 250
no_gaba_inputs = 5
propagation_delay = 2
scales = np.linspace(0.02, 0.200, 30)

### Create structure to hold output
ff_results = pd.DataFrame(columns = ['Scale', 'Inhibition', 'Input', 'ser spks', 'ser ex', 'gaba ex'])

### Perform simulation
cnt = 0
for scale in scales:
    print '{:.1f}%'.format(100 * cnt / scales.shape[0])

    I = scale * distal_in

    # Allocate arrays for spike 5HT spiketrains.
    ser_spkvecs = {
        'ib': np.zeros_like(distal_in),
        'reg': np.zeros_like(distal_in)
    }
    # Create a dict to hold 5HT example traces.
    ser_examples = {}

    # Reps are equivalent to identical 5HT cells in a population.
    for rep in range(no_reps):

        # Iterate over GABA neurons providing input to a single 5HT cell.
        gaba_spkvec = np.zeros_like(distal_in)
        for i in range(no_gaba_inputs):
            t, V, eta, v_T, spks = mGIFs['gaba'].simulate(I, mGIFs['gaba'].El)
            gaba_spkvec += timeToIntVec(spks, distal_in.shape[0] * 0.1, 0.1)

            # Save a sample trace.
            if i == 0 and rep == 0:
                gaba_ex = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}


        # Get total gaba input to a single 5HT cell
        gaba_input = np.convolve(gaba_spkvec, gaba_kernel, 'same')

        # Add in propagation delay.
        prop_delay_int = int(propagation_delay/0.1)
        gaba_input = np.roll(gaba_input, prop_delay_int)
        gaba_input[:prop_delay_int] = 0

        # Try with and without feed-forward inhibition
        for ib_status, ib_multiplier in zip(('ib', 'reg'), (1, 0)):
            t, V, eta, v_T, spks = mGIFs['ser'].simulate(I + gaba_input * ib_multiplier, mGIFs['ser'].El)
            ser_spkvecs[ib_status] += timeToIntVec(spks, distal_in.shape[0] * 0.1, 0.1)

            # Save a sample trace.
            if rep == 0:
                ser_examples[ib_status] = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

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

    cnt += 1

#%%

def PSTH(spktrain, window_width, no_neurons, dt = 0.1):
    """
    Obtain the population firing rate with a resolution of `window_width`.
    """
    kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
    psth = np.convolve(spktrain, kernel, 'same')
    return psth


plt.figure(figsize = (6, 3))

axes = {
    'reg': plt.subplot(131),
    'ib': plt.subplot(132),
    'fi': plt.subplot(133)
}

fi_tmp = []

for i in range(ff_results.shape[0]):

    psth_tmp = PSTH(ff_results.loc[i, 'ser spks'], 0.050, no_reps, 0.0001)
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
axes['fi'].set_ylim(0, 40)

axes['reg'].set_title('No inhibition')
axes['reg'].set_ylabel('Pop. firing rate (Hz)')
axes['reg'].set_xlabel('Time (timesteps)')
axes['reg'].set_xlim(4500, 7000)
axes['reg'].set_ylim(0, 40)
axes['ib'].set_title('Feed forward inhibition')
axes['ib'].set_ylabel('Pop. firing rate (Hz)')
axes['ib'].set_xlabel('Time (timesteps)')
axes['ib'].set_xlim(4500, 7000)
axes['ib'].set_ylim(0, 40)

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'prelim_ff_45.png')

plt.show()
