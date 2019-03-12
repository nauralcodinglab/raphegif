#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./analysis/gating/')


from cell_class import Cell, Recording
import src.pltools as pltools


#%% LOAD DATA

FIGDATA_PATH = './figs/figdata/'
GATING_PATH = './data/gating/'

# Load current steps file
curr_steps = Cell().read_ABF(FIGDATA_PATH + '17n23038.abf')[0]
v_steps = Cell().read_ABF([GATING_PATH + '18619018.abf',
                                      GATING_PATH + '18619019.abf',
                                      GATING_PATH + '18619020.abf'])
v_steps = Recording(np.array(v_steps).mean(axis = 0))

# Load 5CT washon file and extract holding current over time.
# One sweep every 10s?
ser_1A = Cell().read_ABF(FIGDATA_PATH + '18420005.abf')[0]
fct_washon = ser_1A[0, 10000:20000, :].mean(axis = 0)
fct_support = np.arange(0, len(fct_washon)/6, 1/6)

# Import passive membrane parameter data.
params = pd.read_csv('data/DRN_membrane_parameters.csv')

params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)

#%% MAKE FIGURE

IMG_PATH = './figs/ims/thesis/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')


spec = gridspec.GridSpec(3, 4, left = 0.05, height_ratios = [1, 1, 0.6],
                         right = 0.95, top = 0.95, bottom = 0.1,
                         hspace = 0.4)
spec_curr_steps     = gridspec.GridSpecFromSubplotSpec(2, 1, spec[1, 0], height_ratios = [1, 0.2])
spec_v_steps        = gridspec.GridSpecFromSubplotSpec(2, 1, spec[1, 2], height_ratios = [1, 0.2])

hist_color = 'gray'

plt.figure(figsize = (6, 5))

plt.subplot(spec[0, :2])
plt.title('\\textbf{{A}} Long-range connectivity', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[0, 2:])
plt.title('\\textbf{{B}} Identification of 5HT cells', loc = 'left')
pltools.hide_ticks()

Vax = plt.subplot(spec_curr_steps[0, :])
cmdax = plt.subplot(spec_curr_steps[1, :], sharex = Vax)
xlim = (2000, 4500)
Vax.set_title('\\textbf{{C1}} Firing', loc = 'left')
sweeps_to_use = [0, 4, 15]
Vax.set_xlim(xlim)
Vax.plot(
    np.broadcast_to(np.arange(0, curr_steps.shape[1]/10, 0.1)[:, np.newaxis], (curr_steps.shape[1], len(sweeps_to_use))),
    curr_steps[0, :, sweeps_to_use].T - curr_steps[0, 20000:20500, sweeps_to_use].mean(axis = 1),
    'k-', linewidth = 0.5
)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'mV', anchor = (0.6, 0.4), x_size = 500,
    x_label_space = -0.05, y_label_space = -0.05, bar_space = 0, x_on_left = False, ax = Vax
)

cmdax.plot(
    np.broadcast_to(np.arange(0, curr_steps.shape[1]/10, 0.1)[:, np.newaxis], (curr_steps.shape[1], len(sweeps_to_use))),
    curr_steps[1, :, sweeps_to_use].T - curr_steps[1, 20000:20500, sweeps_to_use].mean(axis = 1),
    '-', linewidth = 0.5, color = 'gray'
)
pltools.add_scalebar(y_units = 'pA', y_size = 70, anchor = (0.6, 0.35), omit_x = True, y_label_space = -0.02)

spec_phase = gridspec.GridSpecFromSubplotSpec(2, 2, spec[1, 1], height_ratios = [1, 0.2], width_ratios = [0.2, 1])
plt.subplot(spec_phase[0, 1])
plt.title('\\textbf{{C2}} Phase plot')
start_sweep = 7
t_range = slice(22000, 31500)
plt.plot(
    curr_steps[0, t_range, start_sweep:],
    np.gradient(curr_steps[0, t_range, start_sweep:], axis = 0) / 0.1,
    'k-', alpha = 0.5, linewidth = 0.5
)
plt.ylabel('$\\frac{{dV}}{{dt}}$ (mV/ms)')
plt.xlabel('$V$ (mV)')
pltools.hide_border('tr')

Iax = plt.subplot(spec_v_steps[0, :])
cmdax = plt.subplot(spec_v_steps[1, :], sharex = Iax)
xlim = (2500, 3500)
Iax.set_title('\\textbf{{D}} Currents', loc = 'left')
Iax.set_xlim(xlim)
Iax.set_ylim(-100, 1200)
sweeps_to_use = [0, 4, 8, 12]
Iax.plot(
    np.broadcast_to(np.arange(0, v_steps.shape[1]/10, 0.1)[:, np.newaxis], (v_steps.shape[1], len(sweeps_to_use))),
    v_steps[0, :, sweeps_to_use].T,
    'k-', linewidth = 0.5
)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', anchor = (0.25, 0.5),
    x_size = 250, y_size = 250,
    x_label_space = -0.05, y_label_space = -0.05,
    bar_space = 0, x_on_left = False, ax = Iax
)
plt.text(0.9, 0.9, '+TTX', ha = 'right', va = 'top', transform = Iax.transAxes, size = 'small')

cmdax.plot(
    np.broadcast_to(np.arange(0, v_steps.shape[1]/10, 0.1)[:, np.newaxis], (v_steps.shape[1], len(sweeps_to_use))),
    v_steps[1, :, sweeps_to_use].T,
    '-', linewidth = 0.5, color = 'gray'
)
pltools.hide_ticks(ax = cmdax)
pltools.hide_border(ax = cmdax)


spec_5HT1A = gridspec.GridSpecFromSubplotSpec(2, 2, spec[1, 3], height_ratios = [1, 0.2], width_ratios = [0.2, 1])
plt.subplot(spec_5HT1A[0, 1])
plt.title('\\textbf{{E}} $\mathrm{{5HT_{{1A}}}}$ current', loc = 'left')
plt.plot(fct_support, fct_washon, 'o', color = 'gray', markeredgecolor = 'k', markeredgewidth = 0.7)
plt.plot([3, 8], [60, 60], 'k-', lw = 3)
plt.text(5.5, 62, '5CT', ha = 'center')
plt.ylim(5, 72)
plt.ylabel('$I_{{\mathrm{{hold}}}}$ at $-50$mV (pA)')
plt.xlabel('Time (min)')
pltools.hide_border('tr')


### Passive membrane parameters subplots

# Leak conductance
ax = plt.subplot(spec[2, 0])
plt.title('\\textbf{{F1}} Leak', loc = 'left')
plt.hist(1e3/params_5HT['R'], color = hist_color)
plt.text(
    0.5, 1,
    pltools.p_to_string(stats.shapiro(1e3/params_5HT['R'])[1]),
    ha = 'center', va = 'top', transform = plt.gca().transAxes
)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$g_l$ (pS)')
plt.ylim(0, plt.ylim()[1] * 1.2)

# Capacitance
ax = plt.subplot(spec[2, 1])
plt.title('\\textbf{{F2}} Capacitance', loc = 'left')
plt.hist(params_5HT['C'], color = hist_color)
plt.text(
    0.5, 1,
    pltools.p_to_string(stats.shapiro(params_5HT['C'])[1]),
    ha = 'center', va = 'top', transform = plt.gca().transAxes
)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$C_m$ (pF)')
plt.ylim(0, plt.ylim()[1] * 1.2)

# Membrane time constant
ax = plt.subplot(spec[2, 2])
plt.title('\\textbf{{F3}} Time constant', loc = 'left')
plt.hist(params_5HT['R'] * params_5HT['C'] * 1e-3, color = hist_color)
plt.text(
    0.5, 1,
    pltools.p_to_string(stats.shapiro(params_5HT['R'] * params_5HT['C'] * 1e-3)[1]),
    ha = 'center', va = 'top', transform = plt.gca().transAxes
)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\\tau$ (ms)')
plt.ylim(0, plt.ylim()[1] * 1.2)

# Estimated resting membrane potential
ax = plt.subplot(spec[2, 3])
plt.title('\\textbf{{F4}} Equilibrium $V$', loc = 'left')
plt.hist(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])], color = hist_color)
plt.text(
    0.5, 1,
    pltools.p_to_string(stats.shapiro(params_5HT['El_est'][~np.isnan(params_5HT['El_est'])])[1]),
    ha = 'center', va = 'top', transform = plt.gca().transAxes
)
pltools.hide_border(sides = 'rlt')
plt.yticks([])
plt.xlabel('$\hat{{E}}_l$ (mV)')
plt.ylim(0, plt.ylim()[1] * 1.2)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'physiology.png')

plt.show()


#%% SUMMARY STATISTICS FOR PASSIVE PARAMETERS

print('{:>10}: {:>10.2f} +/- {:>6.2f} GOhm'.format('Rm', np.mean(1e-3 * params_5HT['R']), np.std(1e-3 * params_5HT['R'])))
print('{:>10}: {:>10.2f} +/- {:>6.2f} nS'.format('gl', np.mean(1e3 / params_5HT['R']), np.std(1e3 / params_5HT['R'])))
print('{:>10}: {:>10.1f} +/- {:>6.1f} pF'.format('C', np.mean(params_5HT['C']), np.std(params_5HT['C'])))
print('{:>10}: {:>10.1f} +/- {:>6.1f} ms'.format('tau', np.mean(params_5HT['R'] * params_5HT['C'] * 1e-3), np.std(params_5HT['R'] * params_5HT['C'] * 1e-3)))
print('{:>10}: {:>10.1f} +/- {:>6.1f} mV'.format('El_est', np.nanmean(params_5HT['El_est']), np.nanstd(params_5HT['El_est'])))

labels_ = ['Rm', 'gl', 'C', 'tau', 'El_est']
values_ = [1e-3 * params_5HT['R'],
           1e3 / params_5HT['R'],
           params_5HT['C'],
           params_5HT['R'] * params_5HT['C'] * 1e-3,
           params_5HT['El_est'][~np.isnan(params_5HT['El_est'])]]
for label_, value_ in zip(labels_, values_):
    print('{}: W = {:.5f}, p = {:.5f}'.format(label_, stats.shapiro(value_)[0], stats.shapiro(value_)[1]))

# Fit gamma distribution to membrane time constant
taus_ = params_5HT['R'] * params_5HT['C'] * 1e-3
a, loc, scale = stats.gamma.fit(taus_ / np.median(taus_))
x = np.arange(0, 3, 0.05)
pdf = stats.gamma.pdf(x, a, loc = loc, scale = scale)
sample = np.random.gamma(4.52, 0.19, 63)

plt.figure()
plt.hist(taus_ / np.median(taus_), label = 'Neuronal data')
plt.plot(x, pdf / np.max(pdf) * 16, 'r-', label = 'Fitted distribution')
plt.hist(sample + 0.22, alpha = 0.7, label = 'Artificial data')
plt.legend()
plt.xlabel('Normalized tau')
plt.ylabel('No. of neurons')
plt.show()

print('gamma: a = {:.2f}, loc = {:.2f}, scale = {:.2f}'.format(a, loc, scale))
