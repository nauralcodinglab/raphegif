#%% IMPORT MODULES

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import numpy as np


#%% LOAD DATA AND MODELS

MODEL_PATH = os.path.join('data', 'models')
BENCHMARK_PATH = os.path.join('data', 'processed', 'benchmarks')
SER_EXPT_PATH = os.path.join('data', 'processed', '5HT_fastnoise')
GABA_EXPT_PATH = os.path.join('data', 'processed', 'GABA_fastnoise')

benchmarks = {}
with open(os.path.join(BENCHMARK_PATH, 'serkgifs_benchmark.dat'), 'rb') as f:
    benchmarks['serkgif'] = pickle.load(f)
    f.close()

with open(os.path.join(BENCHMARK_PATH, 'gabagifs_benchmark.dat'), 'rb') as f:
    benchmarks['gabagif'] = pickle.load(f)
    f.close()

models = {}
with open(os.path.join(MODEL_PATH, '5HT', 'serkgifs.lmod'), 'rb') as f:
    models['serkgif'] = pickle.load(f)
    f.close()

with open(os.path.join(MODEL_PATH, 'GABA', 'gaba_gifs.mod'), 'rb') as f:
    models['gabagif'] = pickle.load(f)
    f.close()

experiments = {}
with open(os.path.join(SER_EXPT_PATH, '5HT_goodcells.ldat'), 'rb') as f:
    experiments['ser'] = pickle.load(f)
    f.close()

with open(os.path.join(GABA_EXPT_PATH, 'gaba_goodcells.ldat'), 'rb') as f:
    experiments['gaba'] = pickle.load(f)
    f.close()


#%% MAKE FIGURE

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')
plt.style.use('./figs/scripts/bhrd/poster_mplrc.dms')

ser_ex = 0
gaba_ex = 1

plt.figure(figsize = (16, 6))

spec_outer = gs.GridSpec(1, 2, wspace = 0.5, right = 0.95, left = 0.1)
spec_ser = gs.GridSpecFromSubplotSpec(
    4, 2, spec_outer[:, 0],
    height_ratios = [0.25, 1, 0.4, 0.4], width_ratios = [1., 0.2],
    hspace = 0.05, wspace = 0.5
)
spec_gaba = gs.GridSpecFromSubplotSpec(
    4, 2, spec_outer[:, 1],
    height_ratios = [0.25, 1, 0.4, 0.4], width_ratios = [1., 0.2],
    hspace = 0.05, wspace = 0.5
)

plt.subplot(spec_ser[0, 0])
plt.title(r'\textbf{A1} 5HT sample trace', loc = 'left')
plt.plot(
    experiments['ser'][ser_ex].testset_traces[0].getTime() * 1e-3,
    experiments['ser'][ser_ex].testset_traces[0].I,
    color = 'gray', lw = 1.
)
plt.xticks([])
ser_xlim = plt.xlim()
plt.ylabel('$I$ (nA)')

plt.subplot(spec_ser[1, 0])
plt.plot(
    experiments['ser'][ser_ex].testset_traces[0].getTime() * 1e-3,
    experiments['ser'][ser_ex].testset_traces[0].V,
    color = 'k', lw = 1., label = 'Data'
)
assert experiments['ser'][ser_ex].name == models['serkgif'][ser_ex].name
t, V, _, _, spks = models['serkgif'][ser_ex].simulate(
    experiments['ser'][ser_ex].testset_traces[0].I,
    models['serkgif'][ser_ex].El
)
plt.plot(
    t * 1e-3, V, 'r-', lw = 1, alpha = 0.7, label = '$K$-augmented GIF model'
)
plt.xticks([])
plt.ylabel('$V$ (mV)')
plt.legend(loc = 'upper right')

plt.subplot(spec_ser[2, 0])
data_spks = []
for tr in experiments['ser'][ser_ex].testset_traces:
    data_spks.append(tr.getSpikeTimes() * 1e-3)
plt.eventplot(data_spks, color = 'k')
plt.xticks([])
plt.yticks([])
plt.xlim(ser_xlim)

plt.subplot(spec_ser[3, 0])
model_spks = []
for i in range(9):
    model_spks.append(models['serkgif'][ser_ex].simulate(
        experiments['ser'][ser_ex].testset_traces[0].I,
        models['serkgif'][ser_ex].El
    )[-1] * 1e-3)
plt.eventplot(model_spks, color = 'r')
plt.yticks([])
plt.xlabel('Time (s)')
plt.xlim(ser_xlim)

plt.subplot(spec_ser[:, 1])
plt.title(r'\textbf{A2}', loc = 'left')
plt.ylim(0, 1)
sns.swarmplot(y = benchmarks['serkgif']['Md_vals'], color = 'gray', size = 10, edgecolor = 'k', linewidth = 2.)
plt.xticks([])
plt.ylabel('$M_d^*$')

### GABA NEURONS
plt.subplot(spec_gaba[0, 0])
plt.title(r'\textbf{B1} SOM sample trace', loc = 'left')
plt.plot(
    experiments['gaba'][gaba_ex].testset_traces[0].getTime() * 1e-3,
    experiments['gaba'][gaba_ex].testset_traces[0].I,
    color = 'gray', lw = 1.
)
plt.xticks([])
gaba_xlim = plt.xlim()
plt.ylabel('$I$ (nA)')

plt.subplot(spec_gaba[1, 0])
plt.plot(
    experiments['gaba'][gaba_ex].testset_traces[0].getTime() * 1e-3,
    experiments['gaba'][gaba_ex].testset_traces[0].V,
    color = 'k', lw = 1., label = 'Data'
)
assert experiments['gaba'][gaba_ex].name == models['gabagif'][gaba_ex].name
t, V, _, _, spks = models['gabagif'][gaba_ex].simulate(
    experiments['gaba'][gaba_ex].testset_traces[0].I,
    models['gabagif'][gaba_ex].El
)
plt.plot(
    t * 1e-3, V, 'r-', lw = 1, alpha = 0.7, label = 'GIF model'
)
plt.xticks([])
plt.ylabel('$V$ (mV)')
plt.legend(loc = 'upper right')

plt.subplot(spec_gaba[2, 0])
data_spks = []
for tr in experiments['gaba'][gaba_ex].testset_traces:
    data_spks.append(tr.getSpikeTimes() * 1e-3)
plt.eventplot(data_spks, color = 'k')
plt.xticks([])
plt.yticks([])
plt.xlim(gaba_xlim)

plt.subplot(spec_gaba[3, 0])
model_spks = []
for i in range(9):
    model_spks.append(models['gabagif'][gaba_ex].simulate(
        experiments['gaba'][gaba_ex].testset_traces[0].I,
        models['gabagif'][gaba_ex].El
    )[-1] * 1e-3)
plt.eventplot(model_spks, color = 'r')
plt.yticks([])
plt.xlabel('Time (s)')
plt.xlim(gaba_xlim)

plt.subplot(spec_gaba[:, 1])
plt.title(r'\textbf{B2}', loc = 'left')
plt.ylim(0, 1)
sns.swarmplot(y = benchmarks['gabagif']['Md_vals'], color = 'gray', size = 10, edgecolor = 'k', linewidth = 2.)
plt.xticks([])
plt.ylabel('$M_d^*$')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig7_benchmarks.png'))
