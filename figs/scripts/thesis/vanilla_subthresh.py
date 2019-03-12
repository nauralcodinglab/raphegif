"""
Makes a nice figure comparing linear model on subthreshold dynamics of 5HT and
pyramidal cells. Shows side-by-side sample traces and aggregate
voltage-dependent error.
"""

#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy
import os
import pickle

import sys
sys.path.append('./src')

sys.path.append('analysis/subthresh_mod_selection')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import seaborn as sns
import scipy as sp
import numpy as np

from ModMats import ModMats
from Experiment import Experiment
from SubthreshGIF_K import SubthreshGIF_K
import src.pltools as pltools


#%% DEFINE FUNCTIONS TO GAG VERBOSE POZZORINI METHODS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


#%% INITIALIZE KGIF

KGIF = SubthreshGIF_K(0.1)

KGIF.m_Vhalf = -27
KGIF.m_k = 0.113
KGIF.m_tau = 1.

KGIF.h_Vhalf = -59.9
KGIF.h_k = -0.166
KGIF.h_tau = 50.

KGIF.n_Vhalf = -16.9
KGIF.n_k = 0.114
KGIF.n_tau = 100.

KGIF.E_K = -101.


#%% UNPICKLE DATA

PICKLE_PATH = 'data/mPFC/mPFC_subthresh/'

print 'LOADING DATA'
model_matrices = {
'pyr': [],
'sert': []
}

fnames = [fname for fname in os.listdir(PICKLE_PATH) if fname[-4:].lower() == '.pyc']
for fname in fnames:

    with open(PICKLE_PATH + fname, 'rb') as f:

        modmat_tmp = ModMats(0.1)
        modmat_tmp = pickle.load(f)

    if fname[:3] == 'pyr':
        model_matrices['pyr'].append(modmat_tmp)
    elif fname[:4] == 'sert':
        model_matrices['sert'].append(modmat_tmp)
    else:
        print('Prefix not understood. Skipping.')
        continue

print 'Done!'


#%% FIT MODELS

class CoeffsContainer(object):

    def __init__(self):

        template_dict = {
            'El': [],
            'R': [],
            'C': [],
            'var_explained_dV': []
        }

        self.ohmic_mod = deepcopy(template_dict)

coeff_containers = {
    'pyr': CoeffsContainer(),
    'sert': CoeffsContainer()
}


print 'FITTING MODELS'

for celltype in coeff_containers.keys():
    for i in range(len(model_matrices[celltype])):

        print '\rFitting models to data from cell {}...'.format(i),

        mod = model_matrices[celltype][i]

        mod.setVCutoff(-80)

        ohmic_tmp = mod.fitOhmicMod()

        for key in coeff_containers[celltype].ohmic_mod.keys():
            coeff_containers[celltype].ohmic_mod[key].append(ohmic_tmp[key])


    print 'Done!'


#%% GET PERFORMANCE ON TEST SET

bins = np.arange(-122.5, -26, 5)
bin_centres = (bins[1:] + bins[:-1])/2

print 'GETTING PERFORMANCE ON TEST SET\nWorking',

for celltype in coeff_containers.keys():

    co_cont = coeff_containers[celltype]
    mo_mat = model_matrices[celltype]

    for mod in [co_cont.ohmic_mod]:

        print '.',

        mod['var_explained_Vtest'] = []
        mod['binned_e2_values'] = []
        mod['binned_e2_centres'] = []

        for i in range(len(model_matrices[celltype])):

            KGIF.El = mod['El'][i]
            KGIF.C = mod['C'][i]
            KGIF.gl = 1/mod['R'][i]
            KGIF.gbar_K1 = mod.get('gbar_K1', np.zeros_like(mod['El']))[i]
            KGIF.gbar_K2 = mod.get('gbar_K2', np.zeros_like(mod['El']))[i]

            V_real = deepcopy(mo_mat[i].V_test)
            V_sim = np.empty_like(V_real)

            for sw_ind in range(V_real.shape[1]):

                V_sim[:, sw_ind] = KGIF.simulate(
                mo_mat[i].I_test[:, sw_ind],
                V_real[0, sw_ind]
                )[1]

            model_matrices[celltype][i].V_test_sim = deepcopy(V_sim)

            mod['binned_e2_values'].append(sp.stats.binned_statistic(
                V_real.flatten(), ((V_real - V_sim)**2).flatten(), bins = bins
            )[0])
            mod['binned_e2_centres'].append(bin_centres)

            for sw_ind in range(V_real.shape[1]):
                below_V_cutoff = np.where(V_real[:, sw_ind] < mo_mat[i].VCutoff)[0]
                V_real[below_V_cutoff, sw_ind] = np.nan
                V_sim[below_V_cutoff, sw_ind] = np.nan

            var_explained_Vtest_tmp = (np.nanvar(V_real) - np.nanmedian((V_real - V_sim)**2)) / np.nanvar(V_real)
            mod['var_explained_Vtest'].append(var_explained_Vtest_tmp)

        mod['binned_e2_values'] = np.array(mod['binned_e2_values']).T
        mod['binned_e2_centres'] = np.array(mod['binned_e2_centres']).T

print '\nDone!'

#%% MAKE 5HT FIGURE

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/thesis/'

sert_ex_cell = 1

spec_5HT_outer = gs.GridSpec(1, 3, width_ratios = [1, 1, 0.4], top = 0.9, right = 0.96, bottom = 0.2, left = 0.05, wspace = 0.4)
spec_5HT_tr = gs.GridSpecFromSubplotSpec(2, 1, spec_5HT_outer[:, 0], height_ratios = [0.2, 1], hspace = 0)

plt.figure(figsize = (6, 3))

plt.subplot(spec_5HT_tr[0, :])
plt.title('\\textbf{{A}} 5HT neuron test set trace', loc = 'left')
t_vec = np.arange(
    0,
    model_matrices['sert'][sert_ex_cell].V_test.shape[0] * model_matrices['sert'][sert_ex_cell].dt,
    model_matrices['sert'][sert_ex_cell].dt
)
plt.plot(
    t_vec,
    1e3 * model_matrices['sert'][sert_ex_cell].I_test,
    color = 'gray', linewidth = 0.5
)
plt.xlim(3500, 5800)
pltools.add_scalebar(y_units = 'pA', y_size = 50, omit_x = True, anchor = (0, 0.6), y_label_space = -0.02)

plt.subplot(spec_5HT_tr[1, :])

bins = np.arange(-122.5, -26, 5)[6:]
real = model_matrices['sert'][sert_ex_cell].V_test.mean(axis = 1)
sim = model_matrices['sert'][sert_ex_cell].V_test_sim.mean(axis = 1)
for i in range(1, len(bins)):

    pastel_factor = 0.3
    colour = np.array([0.1, i / len(bins), 0.1]) * (1 - pastel_factor) + pastel_factor

    plt.fill_between(
    t_vec,
    real,
    sim,
    np.logical_and(real >= bins[i - 1], real < bins[i]),
    color = cm.inferno(i/len(bins)), linewidth = 0.5, zorder = 1
    )


plt.axhline(
    -70,
    color = 'k', ls = '--', dashes = (10, 10), lw = 0.5
)
plt.plot(
    t_vec,
    model_matrices['sert'][sert_ex_cell].V_test.mean(axis = 1),
    color = 'k', linewidth = 0.7, label = 'Real neuron'
)
plt.plot(
    t_vec,
    model_matrices['sert'][sert_ex_cell].V_test_sim.mean(axis = 1),
    color = 'r', linewidth = 0.7, label = 'Linear model'
)
plt.text(5200, -69, '$-70$mV', ha = 'center')
plt.text(4250, -50, '+TTX', ha = 'center')
plt.annotate(
    'Model error shaded\naccording to $V$', (4850, -38), ha = 'right', va = 'center',
    xytext = (-10, 20), textcoords = 'offset points',
    arrowprops = {'arrowstyle': '->'}
)
plt.xlim(3500, 5800)
plt.ylim(-85, plt.ylim()[1])
pltools.add_scalebar(y_units = 'mV', x_units = 'ms', anchor = (-0.02, 0.05), x_on_left = False, bar_space = 0, y_label_space = -0.02)
plt.legend(loc = 'lower right')


plt.subplot(spec_5HT_outer[:, 1])
plt.title('\\textbf{{B1}} Binned test set error', loc = 'left')
plt.plot(
    coeff_containers['sert'].ohmic_mod['binned_e2_centres'],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'],
    'k', linewidth = 0.5
)

cc_centres_sub = coeff_containers['sert'].ohmic_mod['binned_e2_centres'][5:, sert_ex_cell]
cc_vals_sub = coeff_containers['sert'].ohmic_mod['binned_e2_values'][5:, sert_ex_cell]
for i in range(len(cc_centres_sub)):

    if i != 5:
        plt.plot(
            cc_centres_sub[i],
            cc_vals_sub[i],
            'o', color = cm.inferno(i / len(cc_centres_sub)), markeredgecolor = 'gray'
        )
    else:
        plt.plot(
            cc_centres_sub[i],
            cc_vals_sub[i],
            'o', color = cm.inferno(i / len(cc_centres_sub)), markeredgecolor = 'gray',
            label = 'Binned error from\nsample trace'
        )
plt.plot(
    np.nanmedian(coeff_containers['sert'].ohmic_mod['binned_e2_centres'], axis = 1),
    np.nanmedian(coeff_containers['sert'].ohmic_mod['binned_e2_values'], axis = 1),
    'k', linewidth = 2, alpha = 0.8, label = 'Median'
)
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlabel('$V$ (mV)')
#plt.ylim(0, plt.ylim()[1])
plt.legend()
pltools.hide_border('tr')


plt.subplot(spec_5HT_outer[:, 2])
plt.title('\\textbf{{B2}}', loc = 'left')
no_neurons = coeff_containers['sert'].ohmic_mod['binned_e2_values'].shape[1]
plt.plot(
    np.array([[0 for i in range(no_neurons)], [1 for i in range(no_neurons)]]),
    np.array([coeff_containers['sert'].ohmic_mod['binned_e2_values'][-7, :],
              coeff_containers['sert'].ohmic_mod['binned_e2_values'][-4, :]]),
    '-', color = 'gray', alpha = 0.5
)
plt.plot(
    [0 for i in range(no_neurons)],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-7, :],
    'o', color = 'gray', markeredgecolor = 'k'
)
plt.plot(
    [1 for i in range(no_neurons)],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-4, :],
    'o', color = 'gray', markeredgecolor = 'k'
)
plt.text(
    0.5, 1,
    pltools.p_to_string(sp.stats.wilcoxon(
        coeff_containers['sert'].ohmic_mod['binned_e2_values'][-7, :],
        coeff_containers['sert'].ohmic_mod['binned_e2_values'][-4, :]
    )[1]),
    ha = 'center', va = 'top', transform = plt.gca().transAxes
)
plt.xticks(
    [0, 1],
    ['Subthreshold\n(${:.0f}$mV)'.format(coeff_containers['sert'].ohmic_mod['binned_e2_centres'][-7, 0]),
    'Perithreshold\n(${:.0f}$mV)'.format(coeff_containers['sert'].ohmic_mod['binned_e2_centres'][-4, 0])],
    rotation = 45
)
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.ylim(plt.ylim()[0], plt.ylim()[1] * 1.1)
plt.xlim(-0.3, 1.3)
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'sert_vanilla_subthresh.png')

plt.show()

# Check for voltage-dependence of model error over -80mV to -45mV range.
# p = 0.00074, N = 14 as of 2018.09.25
sp.stats.friedmanchisquare(*[coeff_containers['sert'].ohmic_mod['binned_e2_values'][i, :] for i in range(-11, -3)])

#%% MAKE PYR FIGURE

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/thesis/'

pyr_ex_cell = 2

spec_pyr_outer = gs.GridSpec(1, 3, width_ratios = [1, 1, 0.4], top = 0.9, right = 0.96, bottom = 0.2, left = 0.05, wspace = 0.4)
spec_pyr_tr = gs.GridSpecFromSubplotSpec(2, 1, spec_pyr_outer[:, 0], height_ratios = [0.2, 1], hspace = 0)

plt.figure(figsize = (6, 3))

plt.subplot(spec_pyr_tr[0, :])
plt.title('\\textbf{{A}} Pyramidal neuron test set trace', loc = 'left')
t_vec = np.arange(
    0,
    model_matrices['pyr'][pyr_ex_cell].V_test.shape[0] * model_matrices['pyr'][pyr_ex_cell].dt,
    model_matrices['pyr'][pyr_ex_cell].dt
)
plt.plot(
    t_vec,
    1e3 * model_matrices['pyr'][pyr_ex_cell].I_test,
    color = 'gray', linewidth = 0.5
)
plt.xlim(3500, 5800)
pltools.add_scalebar(y_units = 'pA', y_size = 50, omit_x = True, anchor = (0, 0.8), y_label_space = -0.02)

plt.subplot(spec_pyr_tr[1, :])
plt.axhline(
    -70,
    color = 'k', ls = '--', dashes = (10, 10), lw = 0.5
)
plt.plot(
    t_vec,
    model_matrices['pyr'][pyr_ex_cell].V_test.mean(axis = 1),
    color = 'k', linewidth = 0.7, label = 'Real neuron'
)
plt.plot(
    t_vec,
    model_matrices['pyr'][pyr_ex_cell].V_test_sim.mean(axis = 1),
    color = 'r', linewidth = 0.7, label = 'Linear model'
)
plt.text(4250, -30, '+TTX', ha = 'center')
plt.xlim(3500, 5800)
pltools.add_scalebar(y_units = 'mV', x_units = 'ms', anchor = (-0.02, 0.05), x_on_left = False, bar_space = 0, y_label_space = -0.02)
plt.legend(loc = 'lower right')


plt.subplot(spec_pyr_outer[:, 1])
plt.title('\\textbf{{B1}} Binned test set error', loc = 'left')
plt.plot(
    coeff_containers['pyr'].ohmic_mod['binned_e2_centres'],
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'],
    'k', linewidth = 0.5
)
plt.plot(
    np.nanmedian(coeff_containers['pyr'].ohmic_mod['binned_e2_centres'], axis = 1),
    np.nanmedian(coeff_containers['pyr'].ohmic_mod['binned_e2_values'], axis = 1),
    'k', linewidth = 2, alpha = 0.8, label = 'Median'
)
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlabel('$V$ (mV)')
plt.legend()
pltools.hide_border('tr')


plt.subplot(spec_pyr_outer[:, 2])
plt.title('\\textbf{{B2}}', loc = 'left')
no_neurons = coeff_containers['pyr'].ohmic_mod['binned_e2_values'].shape[1]
plt.plot(
    np.array([[0 for i in range(no_neurons)], [1 for i in range(no_neurons)]]),
    np.array([coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-7, :],
              coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-4, :]]),
    '-', color = 'gray', alpha = 0.5
)
plt.plot(
    [0 for i in range(no_neurons)],
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-7, :],
    'o', color = 'gray', markeredgecolor = 'k'
)
plt.plot(
    [1 for i in range(no_neurons)],
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-4, :],
    'o', color = 'gray', markeredgecolor = 'k'
)
plt.xticks(
    [0, 1],
    ['Subthreshold\n(${:.0f}$mV)'.format(coeff_containers['pyr'].ohmic_mod['binned_e2_centres'][-7, 0]),
    'Perithreshold\n(${:.0f}$mV)'.format(coeff_containers['pyr'].ohmic_mod['binned_e2_centres'][-4, 0])],
    rotation = 45
)
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlim(-0.3, 1.3)
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'pyr_vanilla_subthresh.png')

plt.show()
