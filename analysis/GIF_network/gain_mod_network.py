#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from scipy import signal

from src.Tools import timeToIntVec

#%% LOAD DATA

with open('./data/simulations/GIF_network/subsample.ldat', 'rb') as f:
    sims = pickle.load(f)
    f.close()

with open('./data/simulations/GIF_network/no_gaba_subsample.ldat', 'rb') as f:
    ng_sims = pickle.load(f)
    f.close()

#%%

ser_spktrain = np.zeros_like(sims['ser_example']['t'])
for spkinds in sims['ser_spktimes']:
    ser_spktrain += timeToIntVec(spkinds, 60000., 0.1)

gaba_spktrain = np.zeros_like(sims['gaba_example']['t'])
for spkinds in sims['gaba_spktimes']:
    gaba_spktrain += timeToIntVec(spkinds, 60000., 0.1)

ng_ser_spktrain = np.zeros_like(ng_sims['ser_example']['t'])
for spkinds in ng_sims['ser_spktimes']:
    ng_ser_spktrain += timeToIntVec(spkinds, 60000., 0.1)

#%%

with open('./data/processed/GIF_network/spktrains_and_examples.dat', 'wb') as f:
    pickle.dump(
        {'ser_spktrain': ser_spktrain,
        'gaba_spktrain': gaba_spktrain,
        'ng_ser_spktrain': ng_ser_spktrain,
        'ser_example': sims['ser_example'],
        'gaba_example': sims['gaba_example'],
        'ng_ser_example': ng_sims['ser_example']},
        f
    )
    f.close()

#%%

def cross_coherence(x, y, fs = 10e3, nperseg = 2**13):
    fxx, pxx = signal.csd(x, x, fs, nperseg = nperseg)
    fyy, pyy = signal.csd(y, y, fs, nperseg = nperseg)
    fxy, pxy = signal.csd(x, y, fs, nperseg = nperseg)

    Cxy = np.abs(pxy)**2/(pxx * pyy)

    return fxy, Cxy

plt.subplot(121)
plt.title('SOM')
plt.semilogx(*cross_coherence(sims['gaba_example']['I'], gaba_spktrain))
plt.xlim(plt.xlim()[0], 1e2)

plt.subplot(122)
plt.title('SER')
plt.semilogx(*cross_coherence(sims['ser_example']['I'], ser_spktrain))
plt.semilogx(*cross_coherence(ng_sims['ser_example']['I'], ng_ser_spktrain), label = 'noff')
plt.legend()
plt.xlim(plt.xlim()[0], 1e2)

#%%

rand_val = 6

plt.semilogx(*cross_coherence(sims['gaba_inmat'][rand_val], sims['gaba_example']['I']))
plt.xlim(plt.xlim()[0], 1e2)

#%%
gaba_spktrain.sum()/(800*60)
ser_spktrain.sum()/(1200*60)

np.zeros(size=(10, 1))
#%% MAKE FIGURE

IMG_PATH = None#os.path.join('figs', 'ims', '2019BHRD')
plt.style.use('./figs/scripts/bhrd/poster_mplrc.dms')

plt.figure(figsize = (16, 10))

spec_outer = gs.GridSpec(3, 2, height_ratios = [1, 0.8, 0.8])
spec_ser = gs.GridSpecFromSubplotSpec(
    2, 1, spec_outer[0, 0],
    height_ratios = [0.2, 1], hspace = 0.05
)
spec_gaba = gs.GridSpecFromSubplotSpec(
    2, 1, spec_outer[0, 1],
    height_ratios = [0.2, 1], hspace = 0.05
)

# 5HT sample trace.
plt.subplot(spec_ser[0, :])
plt.title(r'\textbf{A1} 5HT sample trace', loc = 'left')
plt.plot(
    sims['ser_example']['t'], sims['ser_example']['I'],
    '-', lw = 1., color = 'gray'
)
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec_ser[1, :])
plt.plot(
    sims['ser_example']['t'] * 1e-3, sims['ser_example']['V'],
    '-', lw = 1., color = 'k'
)
plt.xlabel('Time (s)')
plt.ylabel('$V$ (mV)')

# 5HT PSTH.
plt.subplot(spec_outer[1, 0])
plt.title(r'\textbf{A2} 5HT population firing rate (50ms bins)', loc = 'left')
plt.plot(
    sims['ser_example']['t'] * 1e-3,
    np.convolve(ser_spktrain, np.ones(500), 'same') / (1200 * 500e-4),
    color = 'k', ls = 'steps'
)
plt.axhline(
    np.sum(ser_spktrain)/(1200 * 60), color = 'gray',
    ls = '--', lw = 0.7, dashes = (10, 10), label = 'Mean firing rate'
)
plt.ylim(0, plt.ylim()[1])
plt.ylabel('Pop. firing rate (Hz)')
plt.xlabel('Time (s)')
plt.legend()

# 5HT coherence
plt.subplot(spec_outer[2, 0])
plt.title(r'\textbf{A3} 5HT pop. response coherence', loc = 'left')
plt.semilogx(
    *cross_coherence(sims['ser_example']['I'], ser_spktrain),
    color = 'k', label = 'Control'
)
plt.semilogx(
    *cross_coherence(ng_sims['ser_example']['I'], ng_ser_spktrain),
    color = (0.1, 0.1, 0.8), ls = '--', dashes = (10, 7), label = 'SOM neurons removed'
)
plt.ylim(0, plt.ylim()[1])
plt.xlim(plt.xlim()[0], 1e2)
plt.ylabel('Stimulus-response\ncoherence')
plt.xlabel('Frequency (Hz)')
plt.legend()

# GABA sample trace.
plt.subplot(spec_gaba[0, :])
plt.title(r'\textbf{B1} SOM sample trace', loc = 'left')
plt.plot(
    sims['gaba_example']['t'], sims['gaba_example']['I'],
    '-', lw = 1., color = 'gray'
)
plt.xticks([])

plt.subplot(spec_gaba[1, :])
plt.plot(
    sims['gaba_example']['t'] * 1e-3, sims['gaba_example']['V'],
    '-', lw = 1., color = 'k'
)
plt.xlabel('Time (s)')

# 5HT PSTH.
plt.subplot(spec_outer[1, 1])
plt.title(r'\textbf{B2} SOM population firing rate (50ms bins)', loc = 'left')
plt.plot(
    sims['gaba_example']['t'] * 1e-3,
    np.convolve(gaba_spktrain, np.ones(500), 'same') / (800 * 500e-4),
    color = 'k', ls = 'steps'
)
plt.axhline(
    np.sum(gaba_spktrain)/(800 * 60), color = 'gray',
    ls = '--', lw = 0.7, dashes = (10, 10), label = 'Mean firing rate'
)
plt.legend()
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.xlabel('Time (s)')

# 5HT coherence
plt.subplot(spec_outer[2, 1])
plt.title(r'\textbf{B3} SOM pop. response coherence', loc = 'left')
plt.semilogx(
    *cross_coherence(sims['gaba_example']['I'], gaba_spktrain),
    color = 'k'
)
plt.ylim(0., 1.)
plt.xlim(plt.xlim()[0], 1e2)
plt.xlabel('Frequency (Hz)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig8_networksim.png'))
