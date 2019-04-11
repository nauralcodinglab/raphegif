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
sys.path.append('analysis/subthresh_mod_selection')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import scipy as sp
import numpy as np

from ModMats import ModMats
from src.Experiment import Experiment
from src.SubthreshGIF_K import SubthreshGIF_K

import src.pltools as pltools
from src.Tools import gagProcess


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

#%% MAKE FIGURE

plt.rc('text', usetex = True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

IMG_PATH = './figs/ims/TAC3/'


pyr_ex_cell = 2
sert_ex_cell = 1

spec_outer = gs.GridSpec(2, 2)
spec_sert = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[0, 0], height_ratios = [1, 0.2])
spec_pyr = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[0, 1], height_ratios = [1, 0.2])
spec_aggregate = gs.GridSpecFromSubplotSpec(2, 2, spec_outer[1, :], width_ratios = [3, 1], hspace = 0.6, wspace = 0.4)

plt.figure(figsize = (6, 6))

plt.subplot(spec_sert[0, :])
plt.title('\\textbf{{A}} 5HT neuron test set trace')
plt.axhline(
    model_matrices['sert'][sert_ex_cell].VCutoff,
    color = 'k', ls = '--', dashes = (10, 10), lw = 0.5
)
plt.plot(
    t_vec,
    model_matrices['sert'][sert_ex_cell].V_test.mean(axis = 1),
    color = 'k', linewidth = 0.5, label = 'Real neuron'
)
plt.plot(
    t_vec,
    model_matrices['sert'][sert_ex_cell].V_test_sim.mean(axis = 1),
    color = 'r', linewidth = 0.5, alpha = 0.7, label = 'Linear model'
)
pltools.add_scalebar(y_units = 'mV', x_units = 'ms', anchor = (-0.02, 0), x_on_left = False)
plt.legend()

plt.subplot(spec_sert[1, :])
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
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (-0.02, 0.4))

pyr_xlim = (2000, 4000)
plt.subplot(spec_pyr[0, :])
plt.title('\\textbf{{B}} Pyramidal neuron test set trace')
plt.axhline(
    model_matrices['pyr'][pyr_ex_cell].VCutoff,
    color = 'k', ls = '--', dashes = (10, 10), lw = 0.5
)
plt.plot(
    t_vec,
    model_matrices['pyr'][pyr_ex_cell].V_test.mean(axis = 1),
    color = 'k', linewidth = 0.5, label = 'Real neuron'
)
plt.plot(
    t_vec,
    model_matrices['pyr'][pyr_ex_cell].V_test_sim.mean(axis = 1),
    color = 'r', linewidth = 0.5, alpha = 0.7, label = 'Linear model'
)
plt.xlim(pyr_xlim)
plt.ylim(-95, plt.ylim()[1])
pltools.add_scalebar(y_units = 'mV', x_units = 'ms', anchor = (-0.02, 0), x_on_left = False)
plt.legend()

plt.subplot(spec_pyr[1, :])
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
plt.xlim(pyr_xlim)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (-0.02, 0.4))


### Quantified error

plt.subplot(spec_aggregate[:, 0])
plt.title('\\textbf{{C1}} Voltage-dependent test set error', loc = 'left')

plt.plot(
    coeff_containers['sert'].ohmic_mod['binned_e2_centres'],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'],
    'k', linewidth = 0.5
)
plt.plot(
    coeff_containers['pyr'].ohmic_mod['binned_e2_centres'][:, 1:],
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'][:, 1:],
    '--', color = 'gray', linewidth = 0.5
)

plt.plot(
    np.nanmedian(coeff_containers['sert'].ohmic_mod['binned_e2_centres'], axis = 1),
    np.nanmedian(coeff_containers['sert'].ohmic_mod['binned_e2_values'], axis = 1),
    'k', linewidth = 2,
    label = '5HT neurons'
)
plt.plot(
    np.nanmedian(coeff_containers['pyr'].ohmic_mod['binned_e2_centres'][:, 1:], axis = 1),
    np.nanmedian(coeff_containers['pyr'].ohmic_mod['binned_e2_values'][:, 1:], axis = 1),
    '--', color = 'gray', linewidth = 2,
    label = 'Pyramidal neurons'
)

plt.annotate('\\textbf{{C2}}', (-60, 100), ha = 'center')
plt.annotate('\\textbf{{C3}}', (-45, 150), ha = 'center')

plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlabel('$V_m$ (mV)')
plt.ylim(0, plt.ylim()[1])
plt.legend()
pltools.hide_border('tr')


ax = plt.subplot(spec_aggregate[0, 1])
plt.title('\\textbf{{C2}} -60mV', loc = 'left')
x = np.concatenate(
    [[1 for i_ in range(coeff_containers['pyr'].ohmic_mod['binned_e2_values'].shape[1] - 1)],
    [0 for i_ in range(coeff_containers['sert'].ohmic_mod['binned_e2_values'].shape[1])]]
)
y = np.concatenate(
    [coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-7, 1:],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-7, :]
    ]
)
_, p_val = sp.stats.mannwhitneyu(
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-7, 1:],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-7, :],
    alternative = 'two-sided'
)
sns.swarmplot(
    x, y, x,
    palette = ['k', 'gray']
)
plt.text(
    0.5, 0.9, pltools.p_to_string(p_val),
    ha = 'center', va = 'center', transform = plt.gca().transAxes
)
plt.xticks([0, 1], ['5HT', 'Pyr'])
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
ax.legend_.remove()
plt.ylim(plt.ylim()[0], 80)
pltools.hide_border('tr')


ax = plt.subplot(spec_aggregate[1, 1])
plt.title('\\textbf{{C3}} -45mV', loc = 'left')
x = np.concatenate(
    [[1 for i_ in range(coeff_containers['pyr'].ohmic_mod['binned_e2_values'].shape[1] - 1)],
    [0 for i_ in range(coeff_containers['sert'].ohmic_mod['binned_e2_values'].shape[1])]]
)
y = np.concatenate(
    [coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-4, 1:],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-4, :]
    ]
)
_, p_val = sp.stats.mannwhitneyu(
    coeff_containers['pyr'].ohmic_mod['binned_e2_values'][-4, 1:],
    coeff_containers['sert'].ohmic_mod['binned_e2_values'][-4, :],
    alternative = 'two-sided'
)
sns.swarmplot(
    x, y, x,
    palette = ['k', 'gray']
)
plt.text(
    0.5, 0.90, pltools.p_to_string(p_val),
    ha = 'center', va = 'center', transform = plt.gca().transAxes
)
plt.xticks([0, 1], ['5HT', 'Pyr'])
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
ax.legend_.remove()
plt.ylim(plt.ylim()[0], 300)
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'subthresh_celltype_comparison.png', dpi = 300)

plt.show()
