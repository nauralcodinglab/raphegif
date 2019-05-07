#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


#%% LOAD DATA

print 'Loading data...'
DATA_PATH = os.path.join('data', 'processed', 'exclusion')
with open(os.path.join(DATA_PATH, 'experiments.ldat'), 'rb') as f:
    experiments = pickle.load(f)
    f.close()

with open(os.path.join(DATA_PATH, 'intrinsic_reliabilities.ldat'), 'rb') as f:
    reliabilities = pickle.load(f)
    f.close()


#%% GENERATE FIGURE

print 'Generating figure...'

plt.style.use(os.path.join('figs', 'scripts', 'bhrd', 'poster_mplrc.dms'))

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')

# Create dict for example cells.
ex_cells = {
    '5HT': experiments['5HT'][0],
    'mPFC': experiments['mPFC'][0],
    'GABA': experiments['GABA'][0]
}

plt.figure()

spec_outer = gs.GridSpec(2, 3)
spec_sertr = gs.GridSpecFromSubplotSpec(
    3, 1, spec_outer[0, 0],
    hspace = 0.05, height_ratios = [0.2, 1, 0.5]
)
spec_somtr = gs.GridSpecFromSubplotSpec(
    3, 1, spec_outer[0, 1],
    hspace = 0.05, height_ratios = [0.2, 1, 0.5]
)
spec_pyrtr = gs.GridSpecFromSubplotSpec(
    3, 1, spec_outer[0, 2],
    hspace = 0.05, height_ratios = [0.2, 1, 0.5]
)

def plot_sample_traces(spec, title, example_cell, color = 'k'):
    """Standard method for plotting sample testset traces.

    Inputs:
        spec (GridSpec)
            GridSpec object with three rows in which to put plots.
        title (str)
            Title for the uppermost plot.
        example_cell (Experiment)
            Experiment object with traces to plot.
        color
            Color to use for voltage and spikes.
    """

    # Plot input current in first row.
    plt.subplot(spec[0, :])
    plt.title(title, loc = 'left')
    plt.plot(
        example_cell.testset_traces[0].getTime(),
        example_cell.testset_traces[0].I,
        '-', color = 'gray', lw = 0.5
    )
    plt.xticks([])

    # Plot membrane potential (all sweeps) in second row.
    plt.subplot(spec[1, :])
    for tr in example_cell.testset_traces:
        plt.plot(
            tr.getTime(), tr.V, '-', color = color, lw = 0.5, alpha = 0.7
        )
    plt.xticks([])
    xlim_ = plt.xlim()

    # Plot spikes in third row.
    plt.subplot(spec[2, :])
    spktrains = []
    for tr in example_cell.testset_traces:
        spktrains.append(tr.getSpikeTimes())
    plt.eventplot(spktrains, color = color)
    plt.xlim(xlim_)
    plt.yticks([])

### Plot sample traces.
plot_sample_traces(spec_sertr, r'\textbf{A1} 5HT test set', ex_cells['5HT'])
plot_sample_traces(spec_somtr, r'\textbf{B1} SOM test set', ex_cells['GABA'])
plot_sample_traces(spec_pyrtr, r'\textbf{C1} mPFC L5 pyr. test set', ex_cells['mPFC'])

### Plot intrinsic reliabilities.
# Intrinsic reliability of 5HT cells.
plt.subplot(spec_outer[1, 0])
plt.title(r'\textbf{A2} 5HT intrinsic reliability', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(
    reliabilities['supports']['5HT'],
    reliabilities['reliabilities']['5HT'],
    'k-', lw = 0.8, alpha = 0.8
)
plt.ylim(0,1)
plt.ylabel('Intrinsic reliability')
plt.xlabel('Precision (ms)')

# Intrinsic reliability of SOM cells.
plt.subplot(spec_outer[1, 1])
plt.title(r'\textbf{B2} SOM intrinsic reliability', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(
    reliabilities['supports']['GABA'],
    reliabilities['reliabilities']['GABA'],
    'k-', lw = 0.8, alpha = 0.8
)
plt.ylim(0,1)
plt.ylabel('Intrinsic reliability')
plt.xlabel('Precision (ms)')

# Intrinsic reliability of mPFC cells.
plt.subplot(spec_outer[1, 2])
plt.title(r'\textbf{C2} mPFC L5 pyr. intrinsic reliability', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(
    reliabilities['supports']['mPFC'],
    reliabilities['reliabilities']['mPFC'],
    'k-', lw = 0.8, alpha = 0.8
)
plt.ylim(0,1)
plt.ylabel('Intrinsic reliability')
plt.xlabel('Precision (ms)')

plt.tight_layout()

if IMG_PATH is not None:
    print 'Saving figure.'
    plt.savefig(os.path.join(IMG_PATH, 'fig6_testsetstats.png'), dpi = 300)

