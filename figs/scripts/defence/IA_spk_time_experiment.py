#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gs
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./analysis/spk_timing/IA_mod')
sys.path.append('./analysis/spk_timing')
sys.path.append('./figs/scripts')
from grr import pltools
from grr.cell_class import Cell, Recording
import IAmod
from OhmicSpkPredictor import OhmicSpkPredictor, IASpikePredictor, _predict_spk_for_scipy


#%% IMPORT DATA

DATA_PATH = './data/raw/5HT/spk_time/'

"""
inventory = pd.read_csv(DATA_PATH + 'index.csv')
inventory_4AP = inventory.loc[inventory['Cell'] == 'DRN332', :]
inventory_4AP['cumcount'] = inventory_4AP.groupby('Cell').cumcount()
fnames_4AP = inventory_4AP.pivot('Cell', 'cumcount', values = 'Recording')
"""

fnames_baseline = ['18627043.abf', '18627044.abf', '18627045.abf', '18627046.abf', '18627047.abf']
fnames_4AP = ['18627053.abf', '18627054.abf', '18627055.abf']
fnames_wash = ['18627062.abf', '18627063.abf', '18627064.abf']

recs_baseline = Cell().read_ABF([DATA_PATH + fname for fname in  fnames_baseline])
recs_4AP = Cell().read_ABF([DATA_PATH + fname for fname in fnames_4AP])

with open(DATA_PATH + 'predictors_unconstrained.pyc', 'rb') as f:
    predictors = pickle.load(f)

#%% MAKE FIGURE

ga = 10
tau_h = 3
input_strength = 26
Vinput = np.empty((5000, 1))
Vinput[1000:] = input_strength

toy_spk_predictor = IASpikePredictor()
toy_IA_neuron = IAmod.IAmod(ga, tau_h, 0)
toy_IA_neuron.vreset = -60
toy_ohmic_neuron = IAmod.IAmod(0, tau_h, 0)
toy_ohmic_neuron.vreset = -60


IMG_PATH = None#'./figs/ims/defence/'

#plt.style.use('./figs/scripts/defence/defence_mplrc.dms')

spec_mod_outer = gs.GridSpec(
    1, 2,
    left = 0.05, top = 0.85, right = 0.95, bottom = 0.15,
    wspace = 0.5, width_ratios = [1, 0.75]
)
spec_model = gs.GridSpecFromSubplotSpec(2, 2, spec_mod_outer[:, 0], height_ratios = [1, 0.2], hspace = 0.1)

fig = plt.figure(figsize = (5, 2))

### A: simulated proof-of-principle

ax_ohmic = plt.subplot(spec_model[0, 1])
plt.title('Linear model', loc = 'left')
ax_ohmic_I = plt.subplot(spec_model[1, 1])

ax_IA = plt.subplot(spec_model[0, 0])
plt.title('Linear + $I_A$', loc = 'left')
ax_IA_I = plt.subplot(spec_model[1, 0])

for i, V0 in enumerate([-70, -50]):

    Vinput[:1000] = toy_IA_neuron.ss_clamp(V0)
    V_mat_IA, spks_mat, _, _ = toy_IA_neuron.simulate(V0, Vinput)
    V_mat_IA[spks_mat] = 0
    ax_IA.plot(V_mat_IA, 'b-', linewidth = 0.5, alpha = 1/(i + 1))
    ax_IA_I.plot(Vinput, color = 'gray', linewidth = 0.5, alpha = 1/(i + 1))

    Vinput[:1000] = toy_ohmic_neuron.ss_clamp(V0)
    V_mat_ohmic, spks_mat, _, _ = toy_ohmic_neuron.simulate(V0, Vinput)
    V_mat_ohmic[spks_mat] = 0
    ax_ohmic.plot(V_mat_ohmic, 'k-', linewidth = 0.5, alpha = 1/(i + 1))
    ax_ohmic_I.plot(Vinput, color = 'gray', linewidth = 0.5, alpha = 1/(i + 1))

ax_IA.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
ax_IA.annotate('-70mV', (5000, -68), ha = 'right')

ax_ohmic.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
ax_ohmic.annotate('-70mV', (5000, -68), ha = 'right')

ax_IA_I.annotate('26mV', (5000, 24), ha = 'right', va = 'top')
ax_ohmic_I.annotate('26mV', (5000, 24), ha = 'right', va = 'top')

#pltools.add_scalebar(ax = ax_IA, y_units = 'mV', omit_x = True, anchor = (-0.05, 0), y_label_space = (-0.05))
pltools.hide_border(ax = ax_IA)
pltools.hide_ticks(ax = ax_IA)
pltools.hide_border(ax = ax_IA_I)
pltools.hide_ticks(ax = ax_IA_I)

#pltools.add_scalebar(ax = ax_ohmic, y_units = 'mV', omit_x = True, anchor = (-0.05, 0), y_label_space = (-0.05))
pltools.hide_border(ax = ax_ohmic)
pltools.hide_ticks(ax = ax_ohmic)
pltools.hide_border(ax = ax_ohmic_I)
pltools.hide_ticks(ax = ax_ohmic_I)


plt.subplot(spec_mod_outer[:, 1])
#plt.title('\\textbf{{A3}} Effect on spike latency', loc = 'left')
V0_vec = np.linspace(-90, -45)
IA_spk_times = []
ohmic_spk_times = []
for V0 in V0_vec:
    IA_spk_times.append(
        toy_spk_predictor.predict_spk(ga, -45, V0, input_strength, 3, max_time = 10)
    )
    ohmic_spk_times.append(
        toy_spk_predictor.predict_spk(0, -45, V0, input_strength, 3, max_time = 10)
    )

plt.plot(V0_vec, ohmic_spk_times, 'k-', label = 'Linear model')
plt.plot(V0_vec, IA_spk_times, 'b-', label = 'Linear + $I_A$')
plt.ylabel('Spike latency $\\tau_{{\mathrm{{mem}}}}$')
plt.xlabel('$V_{{\mathrm{{pre}}}}$ (mV)')
plt.legend()
pltools.hide_border('tr')


if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_experiment_predictions.png', dpi = 300)

plt.show()


#%%

spec_cell_outer = gs.GridSpec(
    1, 2,
    left = 0.05, top = 0.85, right = 0.95, bottom = 0.2,
    wspace = 0.5, width_ratios = [1, 0.75]
)
spec_4AP = gs.GridSpecFromSubplotSpec(2, 2, spec_cell_outer[:, 0], height_ratios = [1, 0.2], hspace = 0.1)

plt.figure(figsize = (5, 2))

### B: real neurons

trace_time_slice = slice(25400, 28400)
t_vec = np.arange(0, 300, 0.1)
V_ax_bl = plt.subplot(spec_4AP[0, 0])
plt.title(' 5HT neuron', loc = 'left')
plt.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
plt.annotate('-70mV', (300, -72), ha = 'right', va = 'top')
plt.annotate('', (140, 9), (240, 9), arrowprops = {'arrowstyle': '<->'})
plt.text(190, 14, '$\\Delta t_\\mathrm{{spk}}$', ha = 'center')
plt.ylim(-85, 40)

I_ax_bl = plt.subplot(spec_4AP[1, 0])
plt.annotate('30pA', (300, 28), ha = 'right', va = 'top')
pltools.hide_border()
pltools.hide_ticks()

for i, sweep_no in enumerate([3, 8]):
    V_ax_bl.plot(
        t_vec, recs_baseline[0][0, trace_time_slice, sweep_no],
        'b-', lw = 0.5, alpha = 1/(i + 1)
    )
    I_ax_bl.plot(
        t_vec, recs_baseline[0][1, trace_time_slice, sweep_no],
        color = 'gray', lw = 0.5, alpha = 1/(i + 1)
    )

pltools.add_scalebar(
    y_units = 'mV', x_units = 'ms', anchor = (-0.15, 0.5),
    y_label_space = (-0.05), x_on_left = False, x_size = 50,
    bar_space = 0, ax = V_ax_bl
)

V_ax_4AP = plt.subplot(spec_4AP[0, 1])
plt.title('$I_A$ blocked', loc = 'left')
plt.axhline(-70, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
plt.annotate('-70mV', (300, -72), ha = 'right', va = 'top')
plt.annotate('\\textbf{{+4AP}}', (50, 25), ha = 'center')
plt.annotate('', (120, 9), (210, 9), arrowprops = {'arrowstyle': '<->'})
plt.text(165, 14, '$\\Delta t_\\mathrm{{spk}}$', ha = 'center')
plt.ylim(-85, 40)

I_ax_4AP = plt.subplot(spec_4AP[1, 1])
plt.annotate('30pA', (300, 28), ha = 'right', va = 'top')

for i, sweep_no in enumerate([4, 11]):
    V_ax_4AP.plot(
        t_vec, recs_4AP[0][0, trace_time_slice, sweep_no],
        'k-', lw = 0.5, alpha = 1/(i + 1)
    )
    I_ax_4AP.plot(
        t_vec, recs_4AP[0][1, trace_time_slice, sweep_no],
        color = 'gray', lw = 0.5, alpha = 1/(i + 1)
    )

pltools.hide_border(ax = V_ax_4AP)
pltools.hide_ticks(ax = V_ax_4AP)
pltools.hide_border(ax = I_ax_4AP)
pltools.hide_ticks(ax = I_ax_4AP)



latency_dist_ax = plt.subplot(spec_cell_outer[:, 1])
#plt.title('\\textbf{{B3}} Sample latency distribution', loc = 'left')

ex_predictor = predictors[4]['pred_IA']

plt.plot(ex_predictor.V0, ex_predictor.spks, 'k.')

V0_vec_ex = np.linspace(-95, -40)
IA_spk_times_ex = []
for V0 in V0_vec_ex:
    IA_spk_times_ex.append(ex_predictor.predict_spk(
        ex_predictor.gaprime, ex_predictor.thresh, V0,
        ex_predictor.Vinput, ex_predictor.tauh
    ))

plt.plot(
    V0_vec_ex, np.array(IA_spk_times_ex) * ex_predictor.tau,
    'b--', label = 'Linear + $I_A$ fit'
)
plt.annotate('\\textbf{{1}}', (-69, 140))
plt.annotate('\\textbf{{2}}', (-50, 25))
plt.ylabel('Spike latency (ms)')
plt.xlabel('$V_{{\mathrm{{pre}}}}$ (mV)')
plt.legend(loc = 'upper right')
pltools.hide_border('tr')
plt.ylim(-10, 190)

bbox_anchor = (0.05, 0.05, 0.5, 0.4)
ins2 = inset_axes(
    latency_dist_ax, '40%', '100%', loc = 'center right',
    bbox_to_anchor = bbox_anchor, bbox_transform = latency_dist_ax.transAxes
)
ins1 = inset_axes(
    latency_dist_ax, '40%', '100%', loc = 'center left',
    bbox_to_anchor = bbox_anchor, bbox_transform = latency_dist_ax.transAxes
)


ins1.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
ins1.plot(np.arange(0, 195, 0.1), ex_predictor.V[0, 26250:28200, 5], 'k-', lw = 0.5)
ins1.annotate('\\textbf{{1}}', (0, 60))
pltools.hide_border(ax = ins1)
pltools.hide_ticks(ax = ins1)
ins1.set_ylim(-70, 60)

ins2.axhline(-60, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
ins2.plot(np.arange(0, 195, 0.1), ex_predictor.V[0, 26250:28200, 8], 'k-', lw = 0.5)
ins2.annotate('\\textbf{{2}}', (0, 60))
pltools.hide_border(ax = ins2)
pltools.hide_ticks(ax = ins2)
ins2.set_ylim(-70, 60)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_experiment_observed.png', dpi = 300)

plt.show()
