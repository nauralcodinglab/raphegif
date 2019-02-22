"""
2019.02.14
I've optimized the IA inactivation kinetics, restricted gbar_k2 to positive
values, and fitted all coefficients on a restricted set of points.
Now time to see if it all works!
"""

#%% IMPORT MODULES

from __future__ import division

import pickle
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import optimize

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')
sys.path.append('./analysis/regression_tinkering')

from GIF import GIF
from AugmentedGIF import AugmentedGIF
from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from Filter_Exps import Filter_Exps
from SpikeTrainComparator import intrinsic_reliability
from Trace import Trace

import pltools

from model_evaluation import *


#%% READ IN DATA

from load_experiments import experiments

"""with open('./analysis/regression_tinkering/tauh_linesearch_coeffs_new.pyc', 'rb') as f:
    master_coeffs = pickle.load(f)"""

with open(os.path.join('analysis', 'regression_tinkering', 'reference_experiments.pyc'), 'wb') as f:
    pickle.dump(experiments, f)

#%% DEFINE CLASS TO GAG VERBOSE POZZORINI METHODS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


#%% FIT GIFs

GIFs = []

for i, expt in enumerate(experiments):

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = GIF(0.1)

    with gagProcess():

        # Define parameters
        tmp_GIF.Tref = 6.5

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit
        for tr in expt.trainingset_traces:
            tr.setROI([[1000,59000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 14500]])

        tmp_GIF.fit(expt, DT_beforeSpike=1.5)

    GIFs.append(tmp_GIF)
    tmp_GIF.printParameters()

with open(os.path.join('analysis', 'regression_tinkering', 'GIF_reference_mods.pyc'), 'wb') as f:
    pickle.dump(GIFs, f)


#%% FIT OPTIMIZED KGIFs

Opt_KGIFs = []

full_coeffs = []

for i, expt in enumerate(experiments):

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = AugmentedGIF(0.1)

    # Define parameters
    tmp_GIF.Tref = 6.5

    tmp_GIF.eta = Filter_Rect_LogSpaced()
    tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


    tmp_GIF.gamma = Filter_Exps()
    tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

    # Define the ROI of the training set to be used for the fit
    for tr in expt.trainingset_traces:
        tr.setROI([[1000,59000]])
    for tr in expt.testset_traces:
        tr.setROI([[500, 14500]])

    coeffs = []

    for h_tau in np.logspace(np.log2(10), np.log2(150), 10, base = 2):

        print '\rFitting h_tau = {:.1f}ms'.format(h_tau),

        tmp_GIF.h_tau = h_tau

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X, y,
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x'].tolist()

        var_expl_ = var_explained(X, betas, y)

        group_ = h_tau

        row = deepcopy(betas)
        row.extend([group_, var_expl_, expt.name])

        coeffs.append(row)

    coeffs = pd.DataFrame(coeffs)
    coeffs = coeffs.rename({
        coeffs.shape[1] - 3: 'group',
        coeffs.shape[1] - 2: 'var_explained',
        coeffs.shape[1] - 1: 'cell_ID'
        }, axis = 1)

    tmp = convert_betas(coeffs)
    tmp['var_explained'] = coeffs['var_explained']
    tmp['cell_ID'] = coeffs['cell_ID']

    full_coeffs.append(tmp)

    # Assign coeffs of best model
    best_mod_ind = np.argmax(tmp['var_explained'])

    tmp_GIF.C = tmp.loc[best_mod_ind, 'C']
    tmp_GIF.gl = tmp.loc[best_mod_ind, 'gl']
    tmp_GIF.El = tmp.loc[best_mod_ind, 'El']
    tmp_GIF.gbar_K1 = tmp.loc[best_mod_ind, 'gk1']
    tmp_GIF.h_tau = tmp.loc[best_mod_ind, 'group']
    tmp_GIF.gbar_K2 = tmp.loc[best_mod_ind, 'gk2']
    tmp_GIF.eta.setFilter_Coefficients(-np.array(tmp.loc[best_mod_ind, [x for x in tmp.columns if 'AHP' in x]].tolist()))

    # Fit threshold params
    print 'Fitting threshold dynamics.'
    with gagProcess():
        tmp_GIF.fitVoltageReset(expt, tmp_GIF.Tref, False)
        tmp_GIF.fitStaticThreshold(expt)
        tmp_GIF.fitThresholdDynamics(expt)

    Opt_KGIFs.append(tmp_GIF)

print('Done!')

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'wb') as f:
    pickle.dump(Opt_KGIFs, f)

#%%

with open('./analysis/regression_tinkering/Opt_KGIFs.pyc', 'rb') as f:
    Opt_KGIFs = pickle.load(f)

precision = 8.
Md_vals = []
predictions = []

for expt, GIF_, KGIF_ in zip(experiments, GIFs, Opt_KGIFs):

    tmp_Md_vals = []
    tmp_predictions = []

    for mod in [GIF_, KGIF_]:

        with gagProcess():

            # Use the myGIF model to predict the spiking data of the test data set in myExp
            tmp_prediction = expt.predictSpikes(mod, nb_rep=500)

            Md = tmp_prediction.computeMD_Kistler(precision, 0.1)

            tmp_predictions.append(tmp_prediction)
            tmp_Md_vals.append(Md)

    predictions.append(tmp_prediction)
    Md_vals.append(tmp_Md_vals)

    print '{} MD* {}ms: {:.2f}, {:.2f}'.format(expt.name, precision, tmp_Md_vals[0], tmp_Md_vals[1])

with open(os.path.join('analysis', 'regression_tinkering', 'Md_preds.pyc'), 'wb') as f:
    pickle.dump({'Md_vals': Md_vals, 'predictions': predictions}, f)

#%% MAKE PLOT

bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]

group = []
for i in range(len(Md_vals)):
    if i in bad_cells_:
        group.append('bad')
    else:
        group.append('good')

Md_vals_df = pd.DataFrame(Md_vals, columns = ('GIF', 'Optimized KGIF'))
Md_vals_df['group'] = group
del group

IMG_PATH = 'figs/ims/regression_tinkering/opt_KGIF_comparison/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

plt.figure(figsize = (4, 4))
plt.subplot(111)
plt.title('Performance of optimized KGIF')
plt.ylim(0, 1)
sns.swarmplot(
    x = 'Model', y = 'Md*', hue = 'group',
    data = Md_vals_df.melt(
        id_vars = ('group'), var_name = 'Model', value_name = 'Md*'
    ),
    edgecolor = 'gray', linewidth = 0.5, palette = ('k', 'r')
)
plt.plot(
    [[0.2 for i in range(Md_vals_df.shape[0])],
    [0.8 for i in range(Md_vals_df.shape[0])]],
    Md_vals_df.loc[:, ('GIF', 'Optimized KGIF')].T,
    '-', color = 'gray', alpha = 0.7, lw = 2
)
plt.gca().get_legend().set_title('')
plt.ylabel('$M_d^*$ (8ms precision)')
plt.xlabel('')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'performance.png')

plt.show()

#%% STATISTICS ON MD*

good_cells = stats.wilcoxon(
    Md_vals_df.loc[Md_vals_df['group'] == 'good', 'GIF'],
    Md_vals_df.loc[Md_vals_df['group'] == 'good', 'Optimized KGIF']
)
all_cells = stats.wilcoxon(
    Md_vals_df.loc[:, 'GIF'],
    Md_vals_df.loc[:, 'Optimized KGIF']
)
bad_cells = stats.wilcoxon(
    Md_vals_df.loc[Md_vals_df['group'] == 'bad', 'GIF'],
    Md_vals_df.loc[Md_vals_df['group'] == 'bad', 'Optimized KGIF']
)

print 'Good cells: p = {:.5f}'.format(good_cells[1])
print 'Bad cells: p = {:.5f}'.format(bad_cells[1])
print 'All cells: p = {:.5f}'.format(all_cells[1])


#%% PLOT OF COEFFICIENT ESTIMATES

b_GIFs = [GIFs[i] for i in bad_cells_]
g_GIFs = [GIFs[i] for i in range(len(GIFs)) if i not in bad_cells_]
b_Opt_KGIFs = [Opt_KGIFs[i] for i in bad_cells_]
g_Opt_KGIFs = [Opt_KGIFs[i] for i in range(len(GIFs)) if i not in bad_cells_]

def param_corrplot(x_bad, y_bad, x_good, y_good, ax = None):

    if ax is None:
        ax = plt.gca()

    ax.plot(
        x_bad, y_bad,
        'ro', markeredgecolor = 'gray', lw = 0.5, clip_on = False
    )
    ax.plot(
        x_good, y_good,
        'ko', markeredgecolor = 'gray', lw = 0.5, clip_on = False
    )
    ax.set_xlim(
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    )
    ax.set_ylim(
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    )
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k-', zorder = -1, lw = 0.5)


spec_params = gs.GridSpec(2, 3)

plt.figure(figsize = (6, 4))

plt.subplot(spec_params[0, 0])
plt.title(r'R (M$\Omega$)')
param_corrplot(
    [1/x.gl for x in b_GIFs], [1/x.gl for x in b_Opt_KGIFs],
    [1/x.gl for x in g_GIFs], [1/x.gl for x in g_Opt_KGIFs]
)
plt.ylabel('Opt. KGIF')
plt.xlabel('GIF')

plt.subplot(spec_params[0, 1])
plt.title('C (nF)')
param_corrplot(
    [x.C for x in b_GIFs], [x.C for x in b_Opt_KGIFs],
    [x.C for x in g_GIFs], [x.C for x in g_Opt_KGIFs]
)
plt.xlabel('GIF')

plt.subplot(spec_params[0, 2])
plt.title('E (mV)')
param_corrplot(
    [x.El for x in b_GIFs], [x.El for x in b_Opt_KGIFs],
    [x.El for x in g_GIFs], [x.El for x in g_Opt_KGIFs]
)
plt.xlabel('GIF')

plt.subplot(spec_params[1, 0])
plt.title(r'A-type conductance')
plt.ylim(0, 0.025)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.gbar_K1 for x in Opt_KGIFs],
    hue = ['good' if i not in bad_cells_ else 'bad' for i in range(len(Opt_KGIFs))],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.ylabel(r'$\bar{g}_A$ (nS)')
plt.gca().get_legend().remove()
plt.xticks([])

plt.subplot(spec_params[1, 1])
plt.title(r'A-type inactivation time')
plt.ylim(0, 160)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.h_tau for x in Opt_KGIFs],
    hue = ['good' if i not in bad_cells_ else 'bad' for i in range(len(Opt_KGIFs))],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.gca().get_legend().remove()
plt.xticks([])
plt.ylabel(r'$\tau_h$ (ms)')

plt.subplot(spec_params[1, 2])
plt.title(r'Kslow conductance')
plt.ylim(0, 0.01)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.gbar_K2 for x in Opt_KGIFs],
    hue = ['good' if i not in bad_cells_ else 'bad' for i in range(len(Opt_KGIFs))],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.gca().get_legend().remove()
plt.xticks([])
plt.ylabel(r'$\bar{g}_{slow}$ (nS)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'parameter_distributions.png')

plt.show()

#%% SUPPLEMENTARY COEFFICIENT ESTIMATE PLOTS

# Time constant plot
plt.figure(figsize = (2, 2))

plt.subplot(111)
plt.title(r'$\tau_{\mathrm{membrane}}$ (ms)')
param_corrplot(
    [x.C / x.gl for x in b_GIFs], [x.C / x.gl for x in b_Opt_KGIFs],
    [x.C / x.gl for x in g_GIFs], [x.C / x.gl for x in g_Opt_KGIFs]
)
plt.ylabel('Opt. KGIF')
plt.xlabel('GIF')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'time_const.png')

plt.show()


# A-type param plot
plt.figure(figsize = (2, 2))
plt.title(r'A-type param estimates')
plt.plot(
    [x.gbar_K1 for x in b_Opt_KGIFs],
    [x.h_tau for x in b_Opt_KGIFs],
    'ro', markeredgecolor = 'gray', lw = 0.5
)
plt.plot(
    [x.gbar_K1 for x in g_Opt_KGIFs],
    [x.h_tau for x in g_Opt_KGIFs],
    'ko', markeredgecolor = 'gray', lw = 0.5
)
plt.xlabel(r'$\bar{g}_A$ (nS)')
plt.ylabel(r'$\tau_h$ (ms)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'A-type_params.png')

plt.show()

#%% PLOT OF FILTERS

spec_filters = gs.GridSpec(3, 2)

plt.figure()

plt.subplot(spec_filters[0, 0])
plt.title(r'GIF $\eta$')
for mod in b_GIFs:
    x_tmp, y_tmp = mod.eta.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'r-', lw = 0.5)
for mod in g_GIFs:
    x_tmp, y_tmp = mod.eta.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'k-', lw = 0.5)

plt.ylabel('$I$ (nA)')
plt.xlabel('Time (ms)')

plt.subplot(spec_filters[1, 0])
plt.title(r'Optimal KGIF $\eta$')
for mod in b_Opt_KGIFs:
    x_tmp, y_tmp = mod.eta.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'r-', lw = 0.5)
for mod in g_Opt_KGIFs:
    x_tmp, y_tmp = mod.eta.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'k-', lw = 0.5)

plt.ylabel('$I$ (nA)')
plt.xlabel('Time (ms)')

plt.subplot(spec_filters[2, 0])
plt.title(r'Change in $\eta$')
for vanilla, kgif in zip(b_GIFs, b_Opt_KGIFs):
    x_tmp, y_tmp = kgif.eta.getInterpolatedFilter(0.1)
    y_tmp -= vanilla.eta.getInterpolatedFilter(0.1)[1]
    plt.semilogx(x_tmp, y_tmp, 'r-', lw = 0.5)
for vanilla, kgif in zip(g_GIFs, g_Opt_KGIFs):
    x_tmp, y_tmp = kgif.eta.getInterpolatedFilter(0.1)
    y_tmp -= vanilla.eta.getInterpolatedFilter(0.1)[1]
    plt.semilogx(x_tmp, y_tmp, 'k-', lw = 0.5)
plt.axhline(0, color = 'gray', ls = '--', dashes = (10, 10), lw = 0.5)
plt.ylabel(r'$I_{\mathrm{KGIF}} - I_{\mathrm{GIF}}$ (nA)')
plt.xlabel('Time (ms)')

plt.subplot(spec_filters[0, 1])
plt.title(r'GIF $\gamma$')
for mod in b_GIFs:
    x_tmp, y_tmp = mod.gamma.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'r-', lw = 0.5)
for mod in g_GIFs:
    x_tmp, y_tmp = mod.gamma.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'k-', lw = 0.5)

plt.ylabel('Thresh. movement (mV)')
plt.xlabel('Time (ms)')

plt.subplot(spec_filters[1, 1])
plt.title(r'Optimal KGIF $\gamma$')
for mod in b_Opt_KGIFs:
    x_tmp, y_tmp = mod.gamma.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'r-', lw = 0.5)
for mod in g_Opt_KGIFs:
    x_tmp, y_tmp = mod.gamma.getInterpolatedFilter(0.1)
    plt.loglog(x_tmp, y_tmp, 'k-', lw = 0.5)

plt.ylabel('Thresh. movement (mV)')
plt.xlabel('Time (ms)')

plt.subplot(spec_filters[2, 1])
plt.title(r'Change in $\gamma$')
for vanilla, kgif in zip(b_GIFs, b_Opt_KGIFs):
    x_tmp, y_tmp = kgif.gamma.getInterpolatedFilter(0.1)
    y_tmp -= vanilla.gamma.getInterpolatedFilter(0.1)[1]
    plt.semilogx(x_tmp, y_tmp, 'r-', lw = 0.5)
for vanilla, kgif in zip(g_GIFs, g_Opt_KGIFs):
    x_tmp, y_tmp = kgif.gamma.getInterpolatedFilter(0.1)
    y_tmp -= vanilla.gamma.getInterpolatedFilter(0.1)[1]
    plt.semilogx(x_tmp, y_tmp, 'k-', lw = 0.5)
plt.axhline(0, color = 'gray', ls = '--', dashes = (10, 10), lw = 0.5)
plt.ylim(-100, 100)
plt.ylabel(r'KGIF - GIF (mV)')
plt.xlabel('Time (ms)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'spike-triggered filters.png')

plt.show()


#%% COMPARE INTRINSIC RELIABILITY

irel = {
    'empirical': [],
    'GIFs': [],
    'Opt_KGIFs': [],
    'window': [],
    'cell_ID': [],
    'group': []
}

no_reps = 500
windows = [8]


for i, expt, vanilla, kgif in zip([i for i in range(len(experiments))], experiments, GIFs, Opt_KGIFs):

    print '\nComputing intrinsic reliability {:.1f}%'.format(100 * (i + 1) / len(GIFs))

    vanilla_traces = []
    kgif_traces = []

    for j in range(no_reps):
        print '\rGenerating random spikes {:.1f}%'.format(100 * (j + 1)/ no_reps),

        time, V, _, _, spks = vanilla.simulate(expt.testset_traces[0].I, expt.testset_traces[0].V[0])
        tmp = Trace(V, expt.testset_traces[0].I, time[-1], 0.1)
        tmp.spks = (spks / 0.1).astype(np.int32)

        vanilla_traces.append(tmp)

        time, V, _, _, spks = kgif.simulate(expt.testset_traces[0].I, expt.testset_traces[0].V[0])
        tmp = Trace(V, expt.testset_traces[0].I, time[-1], 0.1)
        tmp.spks = (spks / 0.1).astype(np.int32)

        kgif_traces.append(tmp)

    for window in windows:
        irel['cell_ID'].append(expt.name)

        if i in bad_cells_:
            irel['group'].append('bad')
        else:
            irel['group'].append('good')

        irel['window'].append(window)
        irel['empirical'].append(intrinsic_reliability(expt.testset_traces, window))
        irel['GIFs'].append(intrinsic_reliability(vanilla_traces, window))
        irel['Opt_KGIFs'].append(intrinsic_reliability(kgif_traces, window))



irel_df = pd.DataFrame(irel)

#%%

spec_rel = gs.GridSpec(1, 3)

plt.figure(figsize = (6, 2))

plt.suptitle('Intrinsic reliability (8ms)')

plt.subplot(spec_rel[:, 0])
plt.ylim(0, 1)
plt.xlim(0, 1)
param_corrplot(
    irel_df.loc[irel_df['group'] == 'bad', 'empirical'],
    irel_df.loc[irel_df['group'] == 'bad', 'GIFs'],
    irel_df.loc[irel_df['group'] == 'good', 'empirical'],
    irel_df.loc[irel_df['group'] == 'good', 'GIFs'],
)
plt.ylabel('GIF')
plt.xlabel('Data')
plt.gca().set_aspect('equal')

plt.subplot(spec_rel[:, 1])
plt.ylim(0, 1)
plt.xlim(0, 1)
param_corrplot(
    irel_df.loc[irel_df['group'] == 'bad', 'empirical'],
    irel_df.loc[irel_df['group'] == 'bad', 'Opt_KGIFs'],
    irel_df.loc[irel_df['group'] == 'good', 'empirical'],
    irel_df.loc[irel_df['group'] == 'good', 'Opt_KGIFs'],
)
plt.ylabel('Opt. KGIF')
plt.xlabel('Data')
plt.gca().set_aspect('equal')

plt.subplot(spec_rel[:, 2])
plt.ylim(0, 1)
plt.xlim(0, 1)
param_corrplot(
    irel_df.loc[irel_df['group'] == 'bad', 'GIFs'],
    irel_df.loc[irel_df['group'] == 'bad', 'Opt_KGIFs'],
    irel_df.loc[irel_df['group'] == 'good', 'GIFs'],
    irel_df.loc[irel_df['group'] == 'good', 'Opt_KGIFs'],
)
plt.ylabel('Opt. KGIF')
plt.xlabel('GIF')
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.subplots_adjust(top = 0.85)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'intrinsic_reliabilities_8ms.png')

plt.show()
