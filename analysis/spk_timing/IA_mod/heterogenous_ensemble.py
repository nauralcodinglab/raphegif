#%% IMPORT MODULES

from __future__ import division

import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing/IA_mod')
sys.path.append('./figs/scripts')
sys.path.append('./src/')
from IAmod import IAmod, Simulation
import pltools
from Tools import generateOUprocess

#%%

verbose = True

no_neurons = 500
ga_mean = 11
ga_std = 2.77

sigma_noise = 0.5

V0 = -60
Vin_vec = np.linspace(15, 25, 10)
T = 6
T_start = 1
dt = 1e-3

mods = {
    'absent_mod': IAmod(0, 1, sigma_noise),
    'heterogenous_mod': IAmod(np.random.normal(ga_mean, ga_std, no_neurons), 1, sigma_noise),
    'homogenous_mod': IAmod(ga_mean, 1, sigma_noise)
}

sims = {}

for mod_key in mods.keys():

    sims[mod_key] = []

    for Vin in Vin_vec:

        if verbose:
            print('Simulating {} Vin = {}'.format(mod_key, Vin))

        Vin_mat = Vin * np.ones((int(T / dt), no_neurons))
        Vin_mat[:int(T_start / dt)] = 0

        sims[mod_key].append(Simulation(mods[mod_key], V0, Vin_mat, dt))


#%%

dir(Simulation)

sims['homogenous_mod'][-1].t_vec

sims['absent_mod'][-1].simple_plot()
psims['homogenous_mod'][-1].simple_plot()
sims['heterogenous_mod'][-1].simple_plot()
#%%

def PSTH(mod, window_length = 0.2):
    spks_sum = mod.spks.sum(axis = 1)
    kernel = np.ones(int(window_length / mod.dt)) / (window_length * mod.spks.shape[1])

    return np.convolve(spks_sum, kernel, 'same')


IMG_PATH = './figs/ims/thesis/'

window_length = 0.5

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

plt.figure(figsize = (8, 4))

absent_ax = plt.subplot(131)
plt.title('No $I_A$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Time ($\\tau$)')
homogenous_ax = plt.subplot(132)
plt.title('$\\bar{{g}}_A^\prime = 11$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Time ($\\tau$)')
heterogenous_ax = plt.subplot(133)
plt.title('$\\bar{{g}}_A^\prime = 11 \pm 2.77$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Time ($\\tau$)')

for i in range(0, len(sims['homogenous_mod'])):

    absent_ax.plot(sims['absent_mod'][i].t_vec, PSTH(sims['absent_mod'][i], window_length))

    homogenous_ax.plot(sims['homogenous_mod'][i].t_vec, PSTH(sims['homogenous_mod'][i], window_length))

    heterogenous_ax.plot(sims['heterogenous_mod'][i].t_vec, PSTH(sims['heterogenous_mod'][i], window_length))

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'PSTH_IA.png')

plt.show()


#%%

window_length = 0.5

IMG_PATH = './figs/ims/thesis/'

PSTH_at_2 = {}
PSTH_at_2p5 = {}
PSTH_peak = {}

for mod_key in sims.keys():
    PSTH_at_2[mod_key] = []
    PSTH_at_2p5[mod_key] = []
    PSTH_peak[mod_key] = []
    for mod in sims[mod_key]:
        PSTH_tmp = PSTH(mod, window_length)
        PSTH_at_2[mod_key].append(PSTH_tmp[2000])
        PSTH_at_2p5[mod_key].append(PSTH_tmp[2500])
        PSTH_peak[mod_key].append(PSTH_tmp.max())

plt.figure(figsize = (6, 3))

plt.subplot(131)
plt.title('PSTH at 2$\\tau$')
plt.plot(PSTH_at_2['absent_mod'], label = 'No IA')
plt.plot(PSTH_at_2['homogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11$')
plt.plot(PSTH_at_2['heterogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11 \pm 2.77$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Stimulus no.')
plt.legend()

plt.subplot(132)
plt.title('PSTH at 2.5$\\tau$')
plt.plot(PSTH_at_2p5['absent_mod'], label = 'No IA')
plt.plot(PSTH_at_2p5['homogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11$')
plt.plot(PSTH_at_2p5['heterogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11 \pm 2.77$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Stimulus no.')
plt.legend()

plt.subplot(133)
plt.title('PSTH peak')
plt.plot(PSTH_peak['absent_mod'], label = 'No IA')
plt.plot(PSTH_peak['homogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11$')
plt.plot(PSTH_peak['heterogenous_mod'], label = '$\\bar{{g}}_A^\prime = 11 \pm 2.77$')
plt.ylabel('PSTH (spks $\\tau ^{{-1}} \mathrm{{neuron}}^{{-1}}$)')
plt.xlabel('Stimulus no.')
plt.legend()

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'PSTH_IA_over_stim.png')

plt.show()


#%% SUMMARY FIGURE

def ga_to_color(mod, ind = None, lower_bound = 6, upper_bound = 16):

    if ind is None:
        ga = mod.ga
    else:

        try:
            ga = mod.ga[ind]
        except TypeError:
            ga = mod.ga

    fract_ga = (ga - lower_bound) / upper_bound
    fract_ga = min(fract_ga, 1)
    fract_ga = max(fract_ga, 0)
    return (0, 0, fract_ga)

window_length = 0.5
trace_to_plot = 7
neurons_to_plot_trace = 10
neurons_to_plot_raster = 50

spec_outer = gs.GridSpec(4, 1, top = 0.95, right = 0.95, hspace = 0.4, wspace = 0.4, height_ratios = [0.25, 0.01, 1, 1])
spec_hist = gs.GridSpecFromSubplotSpec(1, 3, spec_outer[0, :])
spec_traces = gs.GridSpecFromSubplotSpec(3, 3, spec_outer[2, :], height_ratios = [0.2, 1, 0.6], hspace = 0)
spec_PSTH = gs.GridSpecFromSubplotSpec(2, 3, spec_outer[3, :])

plt.figure()

for i, key in enumerate(['absent_mod', 'homogenous_mod', 'heterogenous_mod']):

    plt.subplot(spec_hist[:, i])
    plt.title(
        '\\textbf{{{}1}} {}'.format(
            ['A', 'B', 'C'][i],
            ['No $I_A$',
             'Homogenous $I_A$',
             'Variable $I_A$'][i]
        ), loc = 'left'
    )
    plt.hist(mods[key].ga, color = 'gray')
    plt.xlim(-1, 20)
    plt.xlabel('$\\bar{{g}}_A$')
    plt.yticks([])
    pltools.hide_border('trl')

    I_ax = plt.subplot(spec_traces[0, i])
    plt.title('\\textbf{{{}2}} Sample traces'.format(['A', 'B', 'C'][i]), loc = 'left')
    pltools.hide_border()
    pltools.hide_ticks()

    V_ax = plt.subplot(spec_traces[1, i])
    pltools.hide_border()
    pltools.hide_ticks()
    raster_ax = plt.subplot(spec_traces[2, i])
    pltools.hide_border()
    pltools.hide_ticks()

    for j in range(max(neurons_to_plot_raster, neurons_to_plot_trace)):

        if j <= neurons_to_plot_trace:
            I_ax.plot(
                Vin_mat[:, 0], color = 'gray', lw = 0.5
            )

            V_ax.plot(
                sims[key][trace_to_plot].V[:, j],
                '-', lw = 0.5, color = ga_to_color(mods[key], j)
            )

        if j <= neurons_to_plot_raster:
            spk_times = np.where(sims[key][trace_to_plot].spks[:, j])[0]
            raster_ax.plot(
                spk_times, [j for k in spk_times],
                '|', markersize = 0.7, color = ga_to_color(mods[key], j)
            )


    raster_ax.set_xlim(0, raster_ax.get_xlim()[1])

    plt.subplot(spec_PSTH[:, i])
    plt.title(
        '\\textbf{{{}3}} PSTH ({} cells)'.format(
            ['A', 'B', 'C'][i],
            sims[key][trace_to_plot].spks.shape[1]
        ), loc = 'left'
    )
    plt.plot(
        sims[key][trace_to_plot].t_vec,
        PSTH(sims[key][trace_to_plot], window_length),
        'k-', lw = 0.8
    )
    pltools.hide_border('tr')
    plt.xlabel('Time ($\\tau_{{mem}}$)')
    if i == 0:
        plt.ylabel('Pop. firing rate (spks neuron$^{{-1}}$ $\\tau_{{mem}}^{{-1}}$)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_hetero_vs_homo_comparison.png')

plt.show()


#%% NOISY INPUT SIMULATIONS

T = 15
burn_in = 3
dt = 1e-3
no_neurons = 500
no_reps = 5
mu = 22

verbose = True

noise_stimuli = []
noise_simulations = {
    'absent_mod': [],
    'homogenous_mod': [],
    'heterogenous_mod': []
}

for rep in range(no_reps):

    noise = generateOUprocess(T - burn_in, 0.5, mu, 2, dt = dt, random_seed = rep * 42)

    Vin_noise_vec = np.concatenate((mu * np.ones(int(burn_in/dt)), noise))
    Vin_noise_mat = np.tile(Vin_noise_vec[:, np.newaxis], (1, no_neurons))

    noise_stimuli.append(Vin_noise_vec)

    for i, key in enumerate(mods.keys()):

        if verbose:
            print('Simulating {} rep {}'.format(key, rep))

        noise_simulations[key].append(Simulation(mods[key], -50, Vin_noise_mat, dt))


#%% NOISY INPUT FIGURE

IMG_PATH = './figs/ims/thesis/'#None

colors = [(0.1, 0.1, 0.1), (0.1, 0.5, 0.1), (0.25, 0.25, 0.8)]

spec_outer = gs.GridSpec(1, 2, width_ratios = [1, 0.2])
spec_psth = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[:, 0], height_ratios = [0.2, 1], hspace = 0.1)
spec_corr = gs.GridSpecFromSubplotSpec(1, 1, spec_outer[:, 1])

plt.figure(figsize = (8, 5))

I_ax = plt.subplot(spec_psth[0, :])
I_ax.set_title(
    '\\textbf{{A}} Population of {} cells encoding a noisy stimulus'.format(no_neurons),
    loc = 'left')
I_ax.plot(
    noise_simulations['absent_mod'][-1].t_vec, noise_stimuli[-1],
    color = 'gray', lw = 0.8
)
I_ax.set_ylabel('Stimulus (mV)')
pltools.hide_border('trb')
I_ax.set_xticks([])

PSTH_ax = plt.subplot(spec_psth[1, :])
PSTH_ax.plot(
    noise_simulations['absent_mod'][-1].t_vec,
    PSTH(noise_simulations['absent_mod'][-1]),
    '-', color = (0.1, 0.1, 0.1), label = 'No $I_A$'
)
PSTH_ax.plot(
    noise_simulations['homogenous_mod'][-1].t_vec,
    PSTH(noise_simulations['homogenous_mod'][-1]),
    '-', color = (0.1, 0.5, 0.1), alpha = 0.7, label = 'Homogenous $I_A$'
)
PSTH_ax.plot(
    noise_simulations['heterogenous_mod'][-1].t_vec,
    PSTH(noise_simulations['heterogenous_mod'][-1]),
    '-', color = (0.25, 0.25, 0.8), alpha = 0.7, label = 'Heterogenous $I_A$'
)
PSTH_ax.legend()
PSTH_ax.set_ylabel('Pop. firing rate (spks neuron$^{{-1}}$ $\\tau_{{mem}}^{{-1}}$)')
PSTH_ax.set_xlabel('Time ($\\tau_{{mem}}$)')
pltools.hide_border('tr')

plt.subplot(spec_corr[:, :])
plt.title('\\textbf{{B}}', loc = 'left')
plt.ylim(0, 1)

r_vals = pd.DataFrame(columns = ['r', 'key'])

for i, key in enumerate(noise_simulations.keys()):


    for rep in range(len(noise_stimuli)):

        psth_tmp = PSTH(noise_simulations[key][rep])[int(burn_in/dt):]
        signal_tmp = noise_stimuli[rep][int(burn_in/dt):]

        df_tmp = pd.DataFrame(
            {'r':stats.pearsonr(psth_tmp, signal_tmp)[0],
             'key':[key]}
        )

        r_vals = r_vals.append(df_tmp)


r_vals = r_vals.reset_index().reindex([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4]).iloc[:, 1:]

sns.swarmplot('key', 'r', data = r_vals, palette = colors, edgecolor = 'gray', linewidth = 0.7)
plt.xticks([])
pltools.hide_border('trb')
plt.ylabel('Stimulus-response correlation $R$')
plt.xlabel('')


if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_noisy_stim_encoding.png')

plt.show()

#%%

spec_tmp = gs.GridSpec(3, 5, hspace = 0.2, wspace = 0)

plt.figure()
for i, key in enumerate(noise_simulations.keys()):

    for rep in range(len(noise_stimuli)):

        plt.subplot(spec_tmp[i, rep])
        plt.title('{} - {}'.format(key[:4], rep + 1))

        plt.xlabel('PSTH')
        plt.ylabel('Signal')

        psth_tmp = PSTH(noise_simulations[key][rep])[int(burn_in/dt):]
        signal_tmp = noise_stimuli[rep][int(burn_in/dt):]

        plt.plot(psth_tmp, signal_tmp, 'k.', alpha = 0.01)

        plt.ylim(15, 28)
        plt.xlim(-0.05, 4)
        plt.xticks([0, 1, 3])

        if i != 2:
            plt.xticks([])
            plt.xlabel('')
        if rep != 0:
            plt.yticks([])
            plt.ylabel('')

plt.tight_layout()

plt.savefig(IMG_PATH + 'IA_PSTH_corr.png')

plt.show()
