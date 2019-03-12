#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy import stats

import sys
sys.path.append('./src')

sys.path.append('./analysis/gls_regression')

from Experiment import *
from AEC_Badel import *
from GIF import *
from AugmentedGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import Filter_Exps

import src.pltools as pltools

from model_evaluation import *

#%% LOAD DATA

from load_experiments import experiments


#%% PLOT VARIABLE DISTRIBUTIONS

IMG_PATH = './figs/ims/var_dists/'
bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]
plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')


def upper_hexbin(x, y, **kwargs):

    plt.hexbin(x, y, **kwargs)
    r, p = stats.pearsonr(x, y)
    plt.text(
        0.05, 0.05, '$r = {:.3f}$'.format(r),
        transform = plt.gca().transAxes, ha = 'left', va = 'bottom'
    )

for i, expt in enumerate(experiments):

    X, y = build_Xy(expt)

    betas = OLS_fit(X, y)
    C = 1./betas[1]
    gk1 = betas[3] * C
    gk2 = betas[4] * C

    noisy_frame = pd.DataFrame(
        {'dV': y, 'V': X[:, 0], 'I': X[:, 1], 'gating1': X[:, 3], 'gating2': X[:, 4]}
    )

    g = sns.PairGrid(noisy_frame.sample(frac = 1, random_state = 42), vars = ('I', 'dV', 'V', 'gating1', 'gating2'))
    g = g.map_diag(sns.distplot, hist_kws=dict(cumulative=True, color = cm.plasma(0)), kde = False)
    g = g.map_lower(plt.hexbin, mincnt = 1, gridsize = 15, linewidth = 0.5, edgecolor = 'k', cmap = cm.plasma)
    g = g.map_upper(upper_hexbin, mincnt = noisy_frame.shape[0]/ 500, gridsize = 15, linewidth = 0.5, edgecolor = 'k', cmap = cm.plasma)
    plt.subplots_adjust(top = 0.95, left = 0.07)
    if i in bad_cells_:
        bad_str = '- excluded'
    else:
        bad_str = ''
    g.fig.suptitle('{} - spike cut (-1.5, 6.5) - gk1 = {:.4f}, gk2 = {:.4f} {}'.format(expt.name, gk1, gk2, bad_str))

    if IMG_PATH is not None:
        plt.savefig(IMG_PATH + '{}_1565spikecut_vardist.png'.format(expt.name))


#%% PLOT PARAMETER DISTRIBUTION

coeffs = {
    'good': [],
    'bad': []
}

for i, expt in enumerate(experiments):

    X, y = build_Xy(expt)
    betas = OLS_fit(X, y)

    if i in bad_cells_:
        coeffs['bad'].append(betas)
    else:
        coeffs['good'].append(betas)

for key in ['good', 'bad']:
    coeffs[key] = pd.DataFrame(coeffs[key])
    coeffs[key]['group'] = key

coeffs = coeffs['good'].append(coeffs['bad'])

coeffs = convert_betas(coeffs)

#%%

"""sns.swarmplot(
    x = 'variable', y = 'value', hue = 'group',
    data = coeffs.loc[:, ['group', 'gk1', 'gk2']].melt(
        id_vars = 'group', value_vars = ['gk1', 'gk2']
    )
)"""

g = sns.jointplot(
    x = 'gk1', y = 'gk2', data = coeffs.loc[coeffs['group'] == 'good', :], kind = 'kde'
)
g.ax_joint.plot(
    coeffs.loc[coeffs['group'] == 'good', 'gk1'],
    coeffs.loc[coeffs['group'] == 'good', 'gk2'],
    'ko', markeredgecolor = 'white'
)
g.ax_joint.plot(
    coeffs.loc[coeffs['group'] == 'bad', 'gk1'],
    coeffs.loc[coeffs['group'] == 'bad', 'gk2'],
    'r.', alpha = 0.6
)
g.fig.set_size_inches(4, 4)
plt.tight_layout()
if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gk_dist_1565spikecut.png')
plt.show()



#%% ILLUSTRATE SPIKE CUT PARAMETERS

spkshape = {
    'x': [],
    'y': []
}

for i, expt in enumerate(experiments):
    x_tmp, y_tmp, _ = expt.trainingset_traces[0].computeAverageSpikeShape()
    spkshape['x'].append(x_tmp)
    spkshape['y'].append(y_tmp)

spkshape['x'] = np.array(spkshape['x']).T
spkshape['y'] = np.array(spkshape['y']).T


plt.figure()
plt.subplot(111)
plt.title('Spike cut parameters in 5HT fast noise data')
plt.plot(
    spkshape['x'][:, bad_cells_],
    spkshape['y'][:, bad_cells_],
    'r-', lw = 0.5
)
plt.plot(
    spkshape['x'][:, [i for i in range(len(experiments)) if i not in bad_cells_]],
    spkshape['y'][:, [i for i in range(len(experiments)) if i not in bad_cells_]],
    'k-', lw = 0.5
)
plt.axvline(4, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5, label = 'Default spike cut (-5, 4)')
plt.axvline(-5, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.axvline(-1.5, color = 'b', ls = '--', dashes = (10, 10), lw = 0.5, label = 'New spike cut (-1.5, 6.5)')
plt.axvline(6.5, color = 'b', ls = '--', dashes = (10, 10), lw = 0.5)
plt.ylabel('$V$ (mV)')
plt.xlabel('Time from detection (ms)')
plt.legend()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'spikecut_params.png')

plt.show()


#%% CROSS VALIDATE BINNED MSE ON TRAINING SET EXCLUDING SPIKES

method = {'base': [3, 4], 'gk1': [4], 'gk2': [3], 'all':[-1]}

cv_binned_error = {}

for key, excl_cols in method.iteritems():

    print('{}'.format(key))

    tmp_error = {
        'train': [],
        'test': []
    }

    for i, expt in enumerate(experiments):

        print '\r{:.1f}%'.format(100*(i + 1)/len(experiments)),

        if i > 5:
            pass

        X, y = build_Xy(expt, excl_cols)

        train_err_tmp, test_err_tmp = cross_validate(X, y, OLS_fit)

        tmp_error['train'].append(train_err_tmp)
        tmp_error['test'].append(test_err_tmp)

    tmp_error['train'] = np.array(tmp_error['train'])
    tmp_error['test'] = np.array(tmp_error['test'])

    cv_binned_error[key] = tmp_error

    print 'Done!'

with open('analysis/gls_regression/cv_dV_error_1565spikecut.pyc', 'wb') as f:
    pickle.dump(cv_binned_error, f)


#%% PLOT OF CROSS-VALIDATED ERROR

with open('analysis/gls_regression/cv_dV_error_1565spikecut.pyc', 'rb') as f:
    cv_binned_error = pickle.load(f)

def cv_plot(error_dict, gridspec_, bad_cells = None):

    if bad_cells is not None:
        good_cells = [i for i in range(error_dict['train'].shape[0]) if i not in bad_cells]
    else:
        good_cells = [i for i in range(error_dict['train'].shape[0])]

    spec = gs.GridSpecFromSubplotSpec(1, 3, gridspec_, wspace = 0.5)

    ax1 = plt.subplot(spec[0, 0])
    #plt.title('CV10 train error', loc = 'left')
    if bad_cells is not None:
        plt.semilogy(
            error_dict['train'][bad_cells, 0, :].T, error_dict['train'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.semilogy(
        error_dict['train'][good_cells, 0, :].T, error_dict['train'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    plt.ylabel(r'Error $\left( \frac{\mathrm{mV}^2}{\mathrm{ms}^2} \right)$')
    plt.xlabel(r'$V_m$ (mV)')

    ax2 = plt.subplot(spec[0, 1])
    plt.title('CV10 test err.', loc = 'left')
    if bad_cells is not None:
        plt.semilogy(
            error_dict['test'][bad_cells, 0, :].T, error_dict['test'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.semilogy(
        error_dict['test'][good_cells, 0, :].T, error_dict['test'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    plt.xlabel(r'$V_m$ (mV)')

    ax3 = plt.subplot(spec[0, 2])
    plt.title('Error ratio', loc = 'left')
    plt.axhline(1, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
    if bad_cells is not None:
        plt.plot(
            error_dict['train'][bad_cells, 0, :].T, error_dict['test'][bad_cells, 1, :].T / error_dict['train'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.plot(
        error_dict['train'][good_cells, 0, :].T, error_dict['test'][good_cells, 1, :].T / error_dict['train'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    #plt.gca().set_yticks([0.9, 0.95, 1, 1.05])
    #plt.gca().set_yticklabels(['$0.90$', '$0.95$', '$1.00$', '$1.05$'])
    plt.ylabel('Test/train error ratio')
    plt.xlabel(r'$V_m$ (mV)')

    return ax1, ax2, ax3

IMG_PATH = './figs/ims/regression_tinkering/'

bad_cells = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

spec = gs.GridSpec(4, 1, hspace = 0.6)

plt.figure(figsize = (6, 8))

for i, key in enumerate(['base', 'gk1', 'gk2', 'all']):
    ax1, _, _ = cv_plot(cv_binned_error[key], spec[i, :], bad_cells)
    ax1.set_title('{} CV10 train err.'.format(key), loc = 'left')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'cv_dV_error_ols_1565spikecut.png')

plt.show()


#%% EVALUDATE RELATIONSHIP BETWEEN BINNED ERROR AND DENSITY

method = {'base': [3, 4], 'gk1': [4], 'gk2': [3], 'all':[-1]}

binned_err_density = {}

for key, excl_cols in method.iteritems():

    print('{}'.format(key))

    tmp_e_d = {
        'bin_centres': [],
        'means': [],
        'densities': []
    }

    for i, expt in enumerate(experiments):

        print '\r{:.1f}%'.format(100*(i + 1)/len(experiments)),

        if i > 5:
            pass

        X, y = build_Xy(expt, excl_cols)

        betas = OLS_fit(X, y)

        bin_centres_, means_, densities_ =  var_explained_binned(X, betas, y, return_proportion = True)

        tmp_e_d['bin_centres'].append(bin_centres_)
        tmp_e_d['means'].append(means_)
        tmp_e_d['densities'].append(densities_)

    tmp_e_d['bin_centres'] = np.array(tmp_e_d['bin_centres'])
    tmp_e_d['means'] = np.array(tmp_e_d['means'])
    tmp_e_d['densities'] = np.array(tmp_e_d['densities'])

    binned_err_density[key] = tmp_e_d

    print 'Done!'
#%%
with open('./analysis/gls_regression/binned_err_densities.pyc', 'wb') as f:
    pickle.dump(binned_err_density, f)

#%%

with open('./analysis/gls_regression/binned_err_densities.pyc', 'rb') as f:
    binned_err_density = pickle.load(f)

plt.figure()
plt.plot(
    binned_err_density['all']['bin_centres'].T,
    binned_err_density['all']['densities'].T,
    'k-'
)
plt.show()

#%%

def rescale_error(error_array):
    error_array = np.copy(error_array)
    error_array -= error_array.min(axis = 0)
    error_array /= error_array.max(axis = 0)
    return error_array


#%%
dens_arr = binned_err_density['all']['densities'].T
means_arr = binned_err_density['all']['means'].T
centres_arr = binned_err_density['all']['bin_centres'].T

plt.figure()

for i in range(dens_arr.shape[1]):

    X_tmp = np.array([1/dens_arr[:, i], np.ones(dens_arr.shape[0])]).T
    y_tmp = means_arr[:, i]
    incl_tmp = np.logical_and(dens_arr[:, i] > 0, means_arr[:, i] > 0)
    incl_tmp[np.argmax(X_tmp[:, 0])] = False

    betas = OLS_fit(X_tmp[incl_tmp, :], y_tmp[incl_tmp])

    x_pred = np.linspace(dens_arr[:, i].min(), dens_arr[:, i].max(), 100)
    X_pred = np.array([1/x_pred, np.ones_like(x_pred)]).T
    y_pred = np.dot(X_pred, betas)

    plt.subplot(5, 5, i + 1)
    if i in bad_cells_:
        titlecolor = 'r'
    else:
        titlecolor = 'k'
    plt.title('{}'.format(experiments[i].name), loc = 'left', color = titlecolor)
    plt.plot(dens_arr[:, i], y_tmp, 'k.')
    plt.plot(x_pred, y_pred, 'r--')
    plt.ylim(0, y_tmp[incl_tmp].max() * 1.1)

    if i % 5 == 0:
        plt.ylabel('MSE')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'MSE_density_plots.png')

plt.show()


#%%

plt.figure()

for i in range(dens_arr.shape[1]):

    plt.subplot(5, 5, i + 1)
    if i in bad_cells_:
        titlecolor = 'r'
    else:
        titlecolor = 'k'
    plt.title('{}'.format(experiments[i].name), loc = 'left', color = titlecolor)
    sns.kdeplot(dens_arr[:, i], centres_arr[:, i])
    plt.plot(dens_arr[:, i], centres_arr[:, i], 'ko', markeredgecolor = 'gray')

    if i % 5 == 0:
        plt.ylabel('$V_m$ (mV)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'dens_voltage.png')

plt.show()


#%%

plt.figure(figsize = (4, 4))
ax = plt.subplot(111)
ax.set_title('Binned voltage distribution all new 5HT cells')
ax.set_color_cycle(sns.color_palette("GnBu_d"))
g = sns.kdeplot(dens_arr.flatten(), centres_arr.flatten(), linewidths = 3, alpha = 0.7)
plt.plot(dens_arr, centres_arr, '-', markeredgecolor = 'gray')
plt.xlim(0, 0.27)
plt.ylim(-87.5, plt.ylim()[1])
plt.ylabel('$V_m$ (mV)')
plt.xlabel(r'Density ($\frac{\mathrm{Pts.\ in\ V\ bin}}{\mathrm{Total\ pts.}}$)')
plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'all_cell_binned_V_density.png')

plt.show()
