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

sys.path.append('./src/')
from IAmod import IAmod, Simulation
import src.pltools as pltools
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
no_neurons = 100
no_reps = 5
mu = 22

verbose = True

ga_mean = 11
ga_std = 2.77

sigma_noise = 0.5

np.random.seed(42)
random_gas = np.random.normal(ga_mean, ga_std, no_neurons)
np.random.seed(43)
random_taus = np.random.gamma(4.52, 0.19, no_neurons) + 0.22

noise_mods = {
    'absent_mod': IAmod(0, 1, sigma_noise),
    'heterogenous_tau_no_IA_mod': IAmod(0, 1, sigma_noise, random_taus),
    'heterogenous_mod': IAmod(random_gas, 1, sigma_noise),
    'homogenous_mod': IAmod(ga_mean, 1, sigma_noise),
    'heterogenous_tau_mod': IAmod(ga_mean, 1, sigma_noise, random_taus),
    'heterogenous_both_mod': IAmod(random_gas, 1, sigma_noise, random_taus)
}

noise_stimuli = []
noise_simulations = {}
for key in noise_mods.keys():
    noise_simulations[key] = []

for rep in range(no_reps):

    noise = generateOUprocess(T - burn_in, 0.5, mu, 2, dt = dt, random_seed = rep * 42)

    Vin_noise_vec = np.concatenate((mu * np.ones(int(burn_in/dt)), noise))
    Vin_noise_mat = np.tile(Vin_noise_vec[:, np.newaxis], (1, no_neurons))

    noise_stimuli.append(Vin_noise_vec)

    for i, key in enumerate(noise_mods.keys()):

        if verbose:
            print('Simulating {} rep {}'.format(key, rep))

        noise_simulations[key].append(Simulation(noise_mods[key], -50, Vin_noise_mat, dt))


#%% NOISY INPUT FIGURE

IMG_PATH = None#'./figs/ims/thesis/'

colors = [(0.1, 0.1, 0.1), (0.1, 0.5, 0.1), (0.25, 0.25, 0.8), (0.8, 0.2, 0.2), (0.5, 0.5, 0.5)]

spec_outer = gs.GridSpec(1, 2, width_ratios = [1, 0.4], bottom = 0.2)
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

labels = {
    'absent_mod': 'No $I_A$',
    'heterogenous_tau_no_IA_mod': 'Hetero. $\\tau$ no $I_A$',
    'homogenous_mod': 'Homo. $I_A$',
    'heterogenous_mod': 'Hetero. $I_A$',
    'heterogenous_tau_mod': 'Hetero. $\\tau$ homo. $I_A$',
    'heterogenous_both_mod': 'Hetero. $\\tau$ hetero. $I_A$'
}
for i, key in enumerate([noise_simulations.keys()[i] for i in [2, 3, 0, 4, 1]]):
    PSTH_ax.plot(
        noise_simulations[key][-1].t_vec,
        PSTH(noise_simulations[key][-1]),
        '-', color = colors[i], label = labels[key], alpha = 0.7
    )

PSTH_ax.legend()
PSTH_ax.set_ylabel('Pop. firing rate (spks neuron$^{{-1}}$ $\\tau_{{mem}}^{{-1}}$)')
PSTH_ax.set_xlabel('Time ($\\tau_{{mem}}$)')
pltools.hide_border('tr')

corr_ax = plt.subplot(spec_corr[:, :])
plt.title('\\textbf{{B}}', loc = 'left')
plt.ylim(0, 1)

r_vals = pd.DataFrame(columns = ['r', 'key'])


for i, key in enumerate(noise_simulations.keys()):


    for rep in range(len(noise_stimuli)):

        psth_tmp = PSTH(noise_simulations[key][rep])[int(burn_in/dt):]
        signal_tmp = noise_stimuli[rep][int(burn_in/dt):]

        df_tmp = pd.DataFrame(
            {'r':stats.pearsonr(psth_tmp, signal_tmp)[0],
             'key':[key],
             'label':[labels[key]]}
        )

        r_vals = r_vals.append(df_tmp)


sns.swarmplot('label', 'r', data = r_vals, palette = colors, edgecolor = 'gray', linewidth = 0.7,
    order = [labels.values()[i] for i in [2, 4, 0, 1, 5, 3]])#order = [labels.values()[i] for i in [2, 3, 0, 4, 1]])
corr_ax.set_xticklabels(corr_ax.get_xticklabels(), rotation = 45, ha = 'right')
pltools.hide_border('tr')
plt.ylabel('Stimulus-response correlation $R$')
plt.xlabel('')


if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'controlled_IA_noisy_stim_encoding.png')

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


#%% CROSS CORRELATION

IMG_PATH = './figs/ims/thesis/'

from scipy import signal

plt.rc('text', usetex = False)

plt.figure()

for i, key in enumerate(noise_simulations.keys()):

    tmp_xcorr = []

    for rep in range(len(noise_stimuli)):

        psth_tmp = PSTH(noise_simulations[key][rep], 0.1)[int(burn_in/dt):]
        signal_tmp = noise_stimuli[rep][int(burn_in/dt):]

        tmp_xcorr.append(signal.correlate(
            psth_tmp - psth_tmp.mean(), signal_tmp - signal_tmp.mean(), mode = 'same') / (len(psth_tmp) * np.std(psth_tmp) * np.std(signal_tmp))
        )

    tmp_xcorr = np.array(tmp_xcorr)

    plt.subplot(2, 3, i + 1)#[3, 5, 2, 1, 6, 4][i])
    plt.title('{}'.format(labels[key]))
    plt.axvline(0, color = 'k', lw = 0.5)
    x = np.arange(-int(tmp_xcorr.shape[1]/2), int(tmp_xcorr.shape[1]/2), 1) * 1e-3
    #plt.plot(np.tile(x[:, np.newaxis], (1, tmp_xcorr.shape[1])), tmp_xcorr.T[:len(x), :], 'k-', lw = 0.5, alpha = 0.5)
    plt.plot(x, tmp_xcorr.mean(axis = 0)[:len(x)], 'r-', label = 'Mean (N = 5)')
    plt.plot(x, np.flip(tmp_xcorr.mean(axis = 0)[:len(x)], -1), '-', color = 'gray', label = 'Flipped mean')
    plt.legend(loc = 'lower right')
    plt.ylim(-0.25, 0.75)
    plt.xlim(-2, 2)

    plt.ylabel('Cross correlation coeff.')
    plt.xlabel('Lag (tau)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'IA_pop_xcorr.png')

plt.show()
