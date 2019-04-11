#%% IMPORT MODULES

from __future__ import division

import os

import copy; from copy import deepcopy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy import stats
from scipy import optimize
import seaborn as sns
import pandas as pd

import sys
sys.path.append('./analysis/regression_tinkering')

from src.Experiment import Experiment
from src.GIF import GIF
from src.resGIF import resGIF
from src.CalciumGIF import CalciumGIF
from src.iGIF_NP import iGIF_NP
from src.iGIF_VR import iGIF_VR
from src.Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from src.Filter_Exps import Filter_Exps
from src.SpikeTrainComparator import intrinsic_reliability
from model_evaluation import *

import src.pltools as pltools
from src.Tools import gagProcess

#%% LOAD DATA

sys.path.append('./figs/scripts/gaba_neurons')
from load_fast_noise import experiments, gagProcess

#%% EXCLUDE BASED ON DRIFT IN NO. SPIKES

drifting_cells = []

for i, expt in enumerate(experiments):
    if len(expt.testset_traces) != 9:
        print('{:>16}Wrong no. of traces. Skipping...'.format(''))
        drifting_cells.append(i) # Exclude it anyway.
        continue

    spks = []
    for j in range(len(expt.testset_traces)):
        spks.append(expt.testset_traces[j].spks)

    no_spks_per_sweep = [len(s_) for s_ in spks]

    r, p = stats.pearsonr(no_spks_per_sweep, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    if p > 0.1:
        stars = ''
    elif p > 0.05:
        stars = 'o'
    elif p > 0.01:
        stars = '*'
    elif p > 0.001:
        stars = '**'
    else:
        stars = '***'

    if np.abs(r) > 0.8:
        drifting_cells.append(i)

    print('{:>2}    {}    R = {:>6.3f}, p = {:>5.3f}   {}'.format(i, expt.name, r, p, stars))

#%% EXCLUDE BASED ON INTRINSIC RELIABILITY

unreliable_cells = []
reliability_ls = []
for i, expt in enumerate(experiments):

    try:
        reliability_tmp = intrinsic_reliability(expt.testset_traces, 8, 0.1)
        reliability_ls.append(reliability_tmp)

        if reliability_tmp < 0.2:
            unreliable_cells.append(i)
            stars = '*'
        else:
            stars = ''

        print('{:>2}    {} IR = {:.3f} {}'.format(
            i, expt.name, reliability_tmp, stars)
        )
    except ValueError:
        print 'Problem with experiment {}'.format(i)

plt.figure()
plt.hist(reliability_ls)
plt.show()

#%%

bad_cell_inds = []
[bad_cell_inds.extend(x) for x in [drifting_cells, unreliable_cells]] #Manually add 7, which has hf noise
bad_cell_inds = np.unique(bad_cell_inds)

bad_cells = []

for i in np.flip(np.sort(bad_cell_inds), -1):
    bad_cells.append(experiments.pop(i))

"""with open(os.path.join('data', 'processed', 'GABA_fastnoise', 'gaba_goodcells.ldat'), 'wb') as f:
    pickle.dump(experiments, f)
    f.close()"""

#%%

t_supp = []
V_mat = []

for expt in experiments:
    t_vec, V_vec, _ = expt.trainingset_traces[0].computeAverageSpikeShape()

    t_supp.append(t_vec)
    V_mat.append(V_vec)

del t_vec, V_vec, _

t_supp = np.array(t_supp).T
V_mat = np.array(V_mat).T

plt.figure()
plt.plot(t_supp, V_mat, 'k-')
plt.axvline(-1.5, color = 'k')
plt.axvline(4, color = 'k')
plt.show()

#%% FIT GIFs

MOD_PATH = os.path.join('data', 'models', 'GABA')

GIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = GIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(expt, DT_beforeSpike=1.5)

    GIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_gifs.mod'), 'wb') as f:
    pickle.dump(GIFs, f)
    f.close()

#%% FIT ResGIFs

resGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = resGIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fitSubthresholdDynamics(expt, DT_beforeSpike=1.5, plot = True, Vmin = -65.)

    resGIFs.append(tmp_GIF)

with open(os.path.join(MOD_PATH, 'gaba_resgifs.mod'), 'wb') as f:
    pickle.dump(resGIFs, f)
    f.close()


#%% FIT CaGIFs

CaGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = CalciumGIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X[mask, :], y[mask],
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x'].tolist()

        #var_expl_ = var_explained(X, betas, y)

        coeffs = pd.DataFrame([betas])
        tmp = convert_betas(coeffs)

        tmp_GIF.C = tmp.loc[0, 'C']
        tmp_GIF.gl = tmp.loc[0, 'gl']
        tmp_GIF.El = tmp.loc[0, 'El']
        tmp_GIF.gbar_K1 = tmp.loc[0, 'gk1']
        tmp_GIF.gbar_K2 = tmp.loc[0, 'gk2']
        tmp_GIF.eta.setFilter_Coefficients(-np.array(tmp.loc[0, [x for x in tmp.columns if 'AHP' in x]].tolist()))

        # Fit threshold dynamics.
        tmp_GIF.fitVoltageReset(expt, tmp_GIF.Tref, False)
        tmp_GIF.fitStaticThreshold(expt)
        tmp_GIF.fitThresholdDynamics(expt)

    CaGIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_cagifs.mod'), 'wb') as f:
    pickle.dump(CaGIFs, f)
    f.close()


#%% FIT iGIFs

iGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = iGIF_NP(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(
            expt, DT_beforeSpike=1.5,
            theta_tau_all = np.logspace(np.log2(1), np.log2(100), 7, base = 2),
            last_bin_constrained = True, do_plot = True
        )

    iGIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_igifs.mod'), 'wb') as f:
    pickle.dump(iGIFs, f)
    f.close()

#%% FIT iGIF_VRs
"""Fit iGIF subclass with variable reset rule.
"""

iGIF_VRs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = iGIF_VR(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(
            expt, DT_beforeSpike=1.5,
            theta_tau_all = np.logspace(np.log2(1), np.log2(100), 7, base = 2),
            do_plot = True
        )

    iGIF_VRs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_igif_vrs.mod'), 'wb') as f:
    pickle.dump(iGIF_VRs, f)
    f.close()

#%% FIT AUGMENTEDGIFs

Opt_KGIFs = []

full_coeffs = []

for i, expt in enumerate(experiments):

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = AugmentedGIF(0.1)

    # Define parameters
    tmp_GIF.Tref = 4.0

    tmp_GIF.eta = Filter_Rect_LogSpaced()
    tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


    tmp_GIF.gamma = Filter_Exps()
    tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

    # Define the ROI of the training set to be used for the fit
    for tr in expt.trainingset_traces:
        tr.setROI([[1000,59000]])
    for tr in expt.testset_traces:
        tr.setROI([[500, 9500]])

    coeffs = []

    for h_tau in np.logspace(np.log2(10), np.log2(300), 10, base = 2):

        print '\rFitting h_tau = {:.1f}ms'.format(h_tau),

        tmp_GIF.h_tau = h_tau

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X[mask, :], y[mask],
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

with open(os.path.join(MOD_PATH, 'gaba_kgifs.mod'), 'wb') as f:
    pickle.dump(Opt_KGIFs, f)
    f.close()

#%% EVALUATE PERFORMANCE

with open('./figs/scripts/gaba_neurons/Opt_KGIFs.pyc', 'rb') as f:
    Opt_KGIFs = pickle.load(f)

precision = 1.
Md_vals = []
predictions = []

for expt, GIF_, KGIF_ in zip(experiments, GIFs, iGIFs):

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

#%%

IMG_PATH = './figs/ims/gaba_cells/'

plt.figure(figsize = (3, 3))
plt.subplot(111)
plt.ylim(0, 1)
sns.swarmplot(
    x = np.array([['GIF', 'iGIF'] for i in Md_vals]).flatten(),
    y = np.array(Md_vals).flatten()
)
plt.plot(np.array([[0.2, 0.8] for i in Md_vals]).T, np.array(Md_vals).T, color = 'gray')
plt.ylabel('$M_d^*$ (1ms precision)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'iGIF_Md_comparison_1ms.png')

plt.show()


#%%


Iweak = np.concatenate((np.zeros(2500), -0.04 * np.ones(1500), 0.02 * np.ones(10000)))
Imed = np.concatenate((np.zeros(2500), -0.04 * np.ones(1500), 0.06 * np.ones(10000)))
Istrong = np.concatenate((np.zeros(2500), -0.04 * np.ones(1500), 0.10 * np.ones(10000)))

spec_outer = gs.GridSpec(len(GIFs), 3)

plt.figure(figsize = (6, 10))

for i, GIF_, iGIF_ in zip([i for i in range(len(GIFs))], GIFs, iGIFs):

    # Perform simulations
    t, V, eta, V_T, spks = GIF_.simulate(Iweak, GIF_.El)
    GIF_weakinput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    t, V, eta, V_T, spks = GIF_.simulate(Imed, GIF_.El)
    GIF_medinput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    t, V, eta, V_T, spks = GIF_.simulate(Istrong, GIF_.El)
    GIF_stronginput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    t, V, eta, V_T, spks = iGIF_.simulate(Iweak, iGIF_.El)
    iGIF_weakinput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    t, V, eta, V_T, spks = iGIF_.simulate(Imed, iGIF_.El)
    iGIF_medinput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    t, V, eta, V_T, spks = iGIF_.simulate(Istrong, iGIF_.El)
    iGIF_stronginput = {'t': t, 'V': V, 'eta': eta, 'V_T': V_T, 'spks': spks}

    # Plot output.
    plt.subplot(spec_outer[i, 0])
    if i == 0:
        plt.title('Weak input')
    plt.plot(GIF_weakinput['t'], GIF_weakinput['V'], 'k-', lw = 0.5)
    plt.plot(iGIF_weakinput['t'], iGIF_weakinput['V'], 'r-', alpha = 0.7, lw = 0.5)
    plt.text(
        0.95, 0.05,
        r'$\Delta M_d^* = {:.3f}$'.format(Md_vals[i][1] - Md_vals[i][0]),
        ha = 'right', va = 'bottom', transform = plt.gca().transAxes
    )
    plt.ylabel('$V$ (mV)')
    if i != len(GIFs) - 1:
        plt.xticks([])
    else:
        plt.xlabel('Time (ms)')

    plt.subplot(spec_outer[i, 1])
    if i == 0:
        plt.title('Medium input')
    plt.plot(GIF_medinput['t'], GIF_medinput['V'], 'k-', lw = 0.5)
    plt.plot(iGIF_medinput['t'], iGIF_medinput['V'], 'r-', alpha = 0.7, lw = 0.5)
    if i != len(GIFs) - 1:
        plt.xticks([])
    else:
        plt.xlabel('Time (ms)')

    plt.subplot(spec_outer[i, 2])
    if i == 0:
        plt.title('Strong input')
    plt.plot(GIF_stronginput['t'], GIF_stronginput['V'], 'k-', lw = 0.5)
    plt.plot(iGIF_stronginput['t'], iGIF_stronginput['V'], 'r-', alpha = 0.7, lw = 0.5)
    if i != len(GIFs) - 1:
        plt.xticks([])
    else:
        plt.xlabel('Time (ms)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'iGIF_firingpattern_comparison.png')

plt.show()

#%% LOAD CURRENT STEPS

"""
Inspect GIF/iGIF spiketrains and compare with data where available.
"""

spec_step_traces = gs.GridSpec(2, 3, height_ratios = [0.2, 1], hspace = 0)

DATA_PATH = os.path.join('data', 'GABA_cells')
fnames_csv = pd.read_csv(os.path.join(DATA_PATH, 'index.csv'))
fnames_csv.columns
curr_step_expts = []

q = 0
for expt, GIF_, iGIF_ in zip(experiments, GIFs, iGIFs):
    fname = fnames_csv.loc[fnames_csv['Cell'] == expt.name, 'Current steps'].tolist()[0]
    if not pd.isnull(fname):
        print 'Curr steps found for {}'.format(expt.name)
        with gagProcess():
            tmp_expt = Experiment(expt.name, 0.1)
            tmp_expt.addTrainingSetTrace(
                'Axon', fname = os.path.join(DATA_PATH, fname),
                I_channel = 1, V_channel = 0
            )
        for tr in tmp_expt.trainingset_traces:
            tr.detectSpikes()
            tr.setROI([[3000, 4200]])
        curr_step_expts.append(tmp_expt)

        plt.figure(figsize = (6, 3))
        plt.suptitle(r'\textbf{{{}}}'.format(expt.name.replace('_', '-')))

        tr_ax = plt.subplot(spec_step_traces[1, 0])
        tr_raster_ax = plt.subplot(spec_step_traces[0, 0])
        gif_ax = plt.subplot(spec_step_traces[1, 1])
        gif_raster_ax = plt.subplot(spec_step_traces[0, 1])
        igif_ax = plt.subplot(spec_step_traces[1, 2])
        igif_raster_ax = plt.subplot(spec_step_traces[0, 2])
        first_spk_sweep = None
        for i, tr in enumerate(tmp_expt.trainingset_traces):
            if tr.getSpikeNbInROI() > 2:

                if first_spk_sweep is None:
                    first_spk_sweep = i

                if (i - first_spk_sweep)%2 == 0 and i - first_spk_sweep < 1:
                    tr_ax.plot(np.arange(0, (len(tr.V) - 0.5) * 0.1, 0.1), tr.V, 'k-', lw = 0.7)
                    tr_raster_ax.plot(tr.getSpikeTimes(), np.zeros_like(tr.getSpikeTimes()), 'k|', markersize = 1.5)

                    for j in range(10):
                        t, V, eta, V_T, spks = GIF_.simulate(tr.I, GIF_.El)
                        gif_raster_ax.plot(spks, j * np.ones_like(spks), 'k|', markersize = 1.5)
                    gif_ax.plot(t, V, 'k-', lw = 0.7)
                    gif_ax.text(0.05, 0.95, '$M_d^* = {:.2f}$'.format(Md_vals[q][0]), transform = gif_ax.transAxes, ha = 'left', va = 'top')

                    for j in range(10):
                        t, V, eta, V_T, spks = iGIF_.simulate(tr.I, iGIF_.El)
                        igif_raster_ax.plot(spks, j * np.ones_like(spks), 'r|', markersize = 1.5)
                    igif_ax.plot(t, V, 'r-', lw = 0.7)
                    igif_ax.text(0.05, 0.95, '$M_d^* = {:.2f}$'.format(Md_vals[q][1]), transform = igif_ax.transAxes, ha = 'left', va = 'top')

            else:
                continue

        tr_raster_ax.set_title('Raw data')
        tr_raster_ax.set_xlim(tr_ax.get_xlim())
        tr_raster_ax.set_yticks([])
        tr_raster_ax.set_xticks([])

        gif_raster_ax.set_title('GIF')
        gif_raster_ax.set_xlim(gif_ax.get_xlim())
        gif_raster_ax.set_yticks([])
        gif_raster_ax.set_xticks([])

        igif_raster_ax.set_title('iGIF')
        igif_raster_ax.set_xlim(igif_ax.get_xlim())
        igif_raster_ax.set_yticks([])
        igif_raster_ax.set_xticks([])

        tr_ax.set_xlabel('Time (ms)')
        tr_ax.set_ylabel('V (mV)')

        gif_ax.set_xlabel('Time (ms)')
        gif_ax.set_ylim(tr_ax.get_ylim())
        igif_ax.set_xlabel('Time (ms)')
        igif_ax.set_ylim(tr_ax.get_ylim())

        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)

        if IMG_PATH is not None:
            plt.savefig(os.path.join(IMG_PATH, '{}_steps_comparison.png'.format(expt.name)))

        plt.show()

    else:
        print 'Curr steps not found for {}'.format(expt.name)

    q += 1



#%%

nb_reps = 30

ISIs = []
sim_ISIs = []

for i, GIF_, iGIF_, expt in zip([i for i in range(len(GIFs))], GIFs, iGIFs, experiments):

    tmp = []
    for tr in expt.trainingset_traces:
        tmp.append(np.diff(tr.getSpikeTimes()))
    ISIs.append(np.concatenate(tmp))
    del tmp

    tmp = {'GIF': [], 'iGIF': []}
    for j in range(nb_reps):
        _, _, _, _, gif_spks = GIF_.simulate(expt.trainingset_traces[0].I, GIF_.El)
        _, _, _, _, igif_spks = iGIF_.simulate(expt.trainingset_traces[0].I, GIF_.El)

        tmp['GIF'].append(np.diff(gif_spks))
        tmp['iGIF'].append(np.diff(igif_spks))

    sim_ISIs_tmp = {}
    for key in tmp.keys():
        sim_ISIs_tmp[key] = np.concatenate(tmp[key])
    del tmp
    sim_ISIs.append(sim_ISIs_tmp)
    print '{:.1f}%'.format(100*(i+1)/len(GIFs))

#%%

def cumdist(x):
    sorted = np.sort(x)
    return sorted, np.cumsum(sorted)/np.sum(x)

bins = np.logspace(np.log2(1), np.log2(5000), 20, base = 2)

for i in range(len(GIFs)):
    plt.figure(figsize = (5, 3))

    plt.suptitle(r'{} - $\Delta M_d^* = {:.3f}$'.format(
        experiments[i].name.replace('_', '-'), Md_vals[i][1] - Md_vals[i][0]
    ))

    plt.subplot(121)
    plt.title('ISI histogram')
    plt.xscale('log')
    plt.hist(ISIs[i], bins = bins, density = True, color = 'k')
    plt.hist(
        sim_ISIs[i]['GIF'], bins = bins, density = True,
        color = 'gray', lw = 3, histtype = 'step'
    )
    plt.hist(
        sim_ISIs[i]['iGIF'], bins = bins, density = True,
        color = 'r', lw = 3, histtype = 'step', alpha = 0.7
    )
    plt.ylabel('Density')
    plt.xlabel('ISI (ms)')

    plt.subplot(122)
    plt.title('Cumulative distribution')
    plt.xscale('log')
    plt.fill_between(
        *cumdist(ISIs[i]),
        color = 'k', clip_on = False, zorder = 10,
        label = 'Training data'
    )
    plt.plot(
        *cumdist(sim_ISIs[i]['GIF']),
        color = 'gray', lw = 3, clip_on = False, zorder = 10,
        label = 'GIF'
    )
    plt.plot(
        *cumdist(sim_ISIs[i]['iGIF']),
        color = 'r', lw = 3, alpha = 0.7, clip_on = False, zorder = 10,
        label = 'iGIF'
    )
    plt.ylim(0, 1)
    plt.legend(loc = 'upper left')
    plt.ylabel('Cumulative fraction')
    plt.xlabel('ISI (ms)')

    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)

    if IMG_PATH is not None:
        plt.savefig(IMG_PATH + '{}_ISI_dist.png'.format(experiments[i].name))

    plt.show()

#%% EXAMINE

Md_arr = np.array(Md_vals)

plt.figure(figsize = (6, 3))

GIF_ax = plt.subplot(131)
plt.title('GIF performance')
plt.xscale('log')
plt.ylabel('Cumulative fraction')
plt.xlabel('ISI (ms)')

iGIF_ax = plt.subplot(132)
plt.title('iGIF performance')
plt.xscale('log')
plt.ylabel('Cumulative fraction')
plt.xlabel('ISI (ms)')

improvement_ax = plt.subplot(133)
plt.title('iGIF improvement')
plt.xscale('log')
plt.ylabel('Cumulative fraction')
plt.xlabel('ISI (ms)')

dmd = Md_arr[:, 1] - Md_arr[:, 0]

for i in range(len(GIFs)):
    GIF_ax.plot(
        *cumdist(ISIs[i]),
        color = ((Md_arr[i, 0] - Md_arr[:, 0].min()) / np.max(Md_arr[:, 0] - Md_arr[:, 0].min()), 0, 0),
        lw = 2, alpha = 0.7
    )

    iGIF_ax.plot(
        *cumdist(ISIs[i]),
        color = ((Md_arr[i, 1] - Md_arr[:, 1].min()) / np.max(Md_arr[:, 1] - Md_arr[:, 1].min()), 0, 0),
        lw = 2, alpha = 0.7
    )

    improvement_ax.plot(
        *cumdist(ISIs[i]),
        color = ((dmd[i] - dmd.min()) / np.max(dmd - dmd.min()), 0, 0),
        lw = 2, alpha = 0.7
    )

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'ISI_cumdist_md_improvement.png')

plt.show()

#%% MAKE FIGURE

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = None#'./figs/ims/gaba_cells/'

ex_cell = 0
xrange = (2000, 9000)

predictions[ex_cell].spks_data
predictions[ex_cell].spks_model

spec_outer = plt.GridSpec(3, 1, height_ratios = [0.2, 1, 0.5])
spec_raster = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[2, :])

plt.figure(figsize = (6, 6))

### Example neuron.
plt.subplot(spec_outer[0, :])
plt.title('\\textbf{{A}} Example trace from positively identified DRN SOM neuron', loc = 'left')
plt.plot(
    experiments[ex_cell].testset_traces[0].getTime(),
    1e3 * experiments[ex_cell].testset_traces[0].I,
    color = 'gray',
    linewidth = 0.5
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (-0.05, 0.4))

plt.subplot(spec_outer[1, :])
plt.plot(
    experiments[ex_cell].testset_traces[0].getTime(),
    experiments[ex_cell].testset_traces[0].V,
    color = 'k', linewidth = 0.5,
    label = 'Real neuron'
)

t, V, _, _, spks = iGIFs[ex_cell].simulate(
    experiments[ex_cell].testset_traces[0].I,
    experiments[ex_cell].testset_traces[0].V[0]
)
V[np.array(spks / 0.1).astype(np.int32)] = 0

plt.plot(
    t, V,
    color = 'r', linewidth = 0.5, alpha = 0.7,
    label = 'Linear model'
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (-0.05, 0.15))

plt.legend()

plt.subplot(spec_raster[0, :])
plt.title('\\textbf{{B}} Spike raster', loc = 'left')
for i, sweep_spks in enumerate(predictions[ex_cell].spks_data):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'k|', markersize = 3
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[1, :])
for i, sweep_spks in enumerate(predictions[ex_cell].spks_model):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'r|', markersize = 3
    )

    if i > len(predictions[ex_cell].spks_data):
        break

plt.xlim(xrange)
pltools.add_scalebar(
    anchor = (0.98, -0.12), x_units = 'ms', omit_y = True,
    x_label_space = -0.08
)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'GABA_good_cell.png', dpi = 300)

plt.show()

print(Opt_KGIFs[ex_cell].gbar_K1)
print(Opt_KGIFs[ex_cell].h_tau)
print(Opt_KGIFs[ex_cell].gbar_K2)

#%%

def param_corrplot(x_good, y_good, ax = None):

    if ax is None:
        ax = plt.gca()

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
    [1/x.gl for x in GIFs], [1/x.gl for x in Opt_KGIFs]
)
plt.ylabel('Opt. KGIF')
plt.xlabel('GIF')

plt.subplot(spec_params[0, 1])
plt.title('C (nF)')
param_corrplot(
    [x.C for x in GIFs], [x.C for x in Opt_KGIFs]
)
plt.xlabel('GIF')

plt.subplot(spec_params[0, 2])
plt.title('E (mV)')
param_corrplot(
    [x.El for x in GIFs], [x.El for x in Opt_KGIFs]
)
plt.xlabel('GIF')

plt.subplot(spec_params[1, 0])
plt.title(r'A-type conductance')
#plt.ylim(0, 0.025)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.gbar_K1 for x in Opt_KGIFs],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.ylabel(r'$\bar{g}_A$ (nS)')
plt.xticks([])

plt.subplot(spec_params[1, 1])
plt.title(r'A-type inactivation time')
#plt.ylim(0, 160)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.h_tau for x in Opt_KGIFs],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
plt.ylabel(r'$\tau_h$ (ms)')

plt.subplot(spec_params[1, 2])
plt.title(r'Kslow conductance')
#plt.ylim(0, 0.01)
sns.swarmplot(
    x = ['tmp' for i in range(len(Opt_KGIFs))],
    y = [x.gbar_K2 for x in Opt_KGIFs],
    palette = ('k', 'r'), clip_on = False, edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
plt.ylabel(r'$\bar{g}_{slow}$ (nS)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'parameter_distributions.png')

plt.show()

#%% MD DISTRIBUTION FIGURE

plt.figure(figsize = (1.2, 3))

plt.subplot(111)
plt.ylim(0, 1)
sns.swarmplot(x = [0 for i in Md_vals], y = Md_vals, facecolor = 'black', edgecolor = 'gray', linewidth = 0.5)
pltools.hide_border('trb')
plt.xticks([])
plt.ylabel('Md* (4ms)')

plt.tight_layout()

plt.savefig(IMG_PATH + 'gaba_md_distribution.png', dpi = 300)

plt.show()


#%%

gl_ls = []
C_ls = []
El_ls = []
VT_ls = []
DV_ls = []
spks_ls = []

for expt, GIF_ in zip(experiments, GIFs):

    gl_ls.append(GIF_.gl)
    C_ls.append(GIF_.C)
    El_ls.append(GIF_.El)
    VT_ls.append(GIF_.Vt_star)
    DV_ls.append(GIF_.DV)

    spks_ls.append(expt.getTrainingSetNbOfSpikes())


plt.figure()

plt.suptitle('DRN SOM neuron vanilla GIF fit and extracted parameters')

spec = gs.GridSpec(2, 3)

gl_ax   = plt.subplot(spec[0, 0])
plt.ylabel('gl')
plt.xlabel('Md*')
plt.xlim(0, 1)
C_ax    = plt.subplot(spec[0, 1])
plt.ylabel('C')
plt.xlabel('Md*')
plt.xlim(0, 1)
El_ax   = plt.subplot(spec[0, 2])
plt.ylabel('El (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
VT_ax   = plt.subplot(spec[1, 0])
plt.ylabel('Threshold (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
DV_ax   = plt.subplot(spec[1, 1])
plt.ylabel('$\Delta V$ (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
spks_ax = plt.subplot(spec[1, 2])
plt.ylabel('No. spikes in training set')
plt.xlabel('Md*')
plt.xlim(0, 1)

for i in range(len(gl_ls)):

    gl_ax.plot(Md_vals[i], gl_ls[i], 'ko', alpha = 0.7)
    C_ax.plot(Md_vals[i], C_ls[i], 'ko', alpha = 0.7)
    El_ax.plot(Md_vals[i], El_ls[i], 'ko', alpha = 0.7)
    VT_ax.plot(Md_vals[i], VT_ls[i], 'ko', alpha = 0.7)
    DV_ax.plot(Md_vals[i], DV_ls[i], 'ko', alpha = 0.7)
    spks_ax.plot(Md_vals[i], spks_ls[i], 'ko', alpha = 0.7)

plt.tight_layout()
plt.subplots_adjust(top = 0.9)

plt.savefig(IMG_PATH + 'GABA_fit_and_parameters.png', dpi = 300)

plt.show()


#%%

plt.figure()

plt.subplot(111)
plt.title('$\gamma$')
plt.axhline(0, ls = '--', color = 'k', lw = 0.5, dashes = (10, 10))

for i in range(len(GIFs)):
    x, y = GIFs[i].gamma.getInterpolatedFilter(0.1)
    plt.loglog(x, y, 'k-')

plt.ylim(plt.ylim()[0], 30)
plt.ylabel('Threshold movement (mV)')
plt.xlabel('Time (ms)')

plt.savefig(IMG_PATH + 'GABA_gamma.png', dpi = 300)

plt.show()


#%%

plt.figure()

plt.subplot(111)
plt.title('$\gamma$-filter first bin amplitude vs. fit')
plt.ylabel('$\gamma$ first bin')
plt.xlabel('Md* (4ms)')
plt.xlim(0, 1)

for i in range(len(GIFs)):
    plt.plot(Md_vals[i], GIFs[i].gamma.getCoefficients()[0], 'ko', markersize = 10, alpha = 0.7)

plt.savefig(IMG_PATH + 'GABA_gamma_vs_fit.png', dpi = 300)

plt.show()

#%%

def compute_ISIs(trace):

    spk_times = trace.getSpikeTimes()

    ISIs = np.diff(spk_times)

    return ISIs

plt.figure()

no_bins = 35
plt.hist(compute_ISIs(experiments[-1].trainingset_traces[0]), color = 'k', label = 'Reference cell 1', bins = no_bins)
plt.hist(compute_ISIs(experiments[-2].trainingset_traces[0]), color = 'gray', alpha = 0.5, label = 'Reference cell 2', bins = no_bins)
plt.hist(compute_ISIs(experiments[0].trainingset_traces[0]), color = 'r', alpha = 0.5, label = 'Worst cell', bins = no_bins)
plt.legend()

pltools.hide_border('trl')
plt.yticks([])
plt.xlabel('ISI (ms)')

plt.savefig(IMG_PATH + 'GABA_ISI_distribution.png', dpi = 300)

plt.show()

#%%

violin_df = pd.DataFrame(columns = ('Label', 'ISI'))

for cell_no in range(len(experiments)):

    ISIs_tmp = []
    for tr_no in range(3):
        ISIs_tmp.extend(compute_ISIs(experiments[cell_no].trainingset_traces[tr_no]))


    name_vec_tmp = ['{}\nMd = {:.2f}'.format(experiments[cell_no].name.replace('_', ' '), Md_vals[cell_no]) for i in range(len(ISIs_tmp))]

    if cell_no == 0:
        violin_df = pd.DataFrame({'Label': name_vec_tmp, 'ISI': ISIs_tmp})
    else:
        violin_df = violin_df.append(pd.DataFrame({'Label': name_vec_tmp, 'ISI': ISIs_tmp}))


plt.figure(figsize = (8, 8))

violin_ax = plt.subplot(111)
plt.title('ISI distribution of DRN SOM neurons')
sns.violinplot(x = 'ISI', y = 'Label', data = violin_df, cut = 0, bw = 0.1)
plt.ylabel('')
plt.xlabel('ISI (ms)')
plt.xlim(0, plt.xlim()[1])


plt.savefig(IMG_PATH + 'ISI_violin.png', dpi = 300)

plt.show()
