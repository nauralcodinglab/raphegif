"""
SUBTHRESHOLD GIF ANALYSIS
"""

#%% IMPORT MODULES

from __future__ import division
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import itertools

# Import GIF toolbox modules from read-only clone
import sys
sys.path.append('src')

from Experiment import Experiment
from SubthreshGIF import SubthreshGIF
from SubthreshGIF_K import SubthreshGIF_K
from AEC_Badel import AEC_Badel
from Trace import Trace
import Tools


#%% DEFINE FUNCTIONS TO GAG VERBOSE POZZORINI METHODS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout



#%% LOAD DATA

path = ('data/subthreshold_expts/')

cells = [['c0_AEC_18125000.abf', 'c0_Train_18125001.abf', 'c0_Test_18125002.abf'],
         ['c1_AEC_18125011.abf', 'c1_Train_18125012.abf', 'c1_Test_18125013.abf'],
         ['c2_AEC_18125026.abf', 'c2_Train_18125027.abf', 'c2_Test_18125028.abf'],
         ['c3_AEC_18126000.abf', 'c3_Train_18126001.abf', 'c3_Test_18126002.abf'],
         ['c4_AEC_18126009.abf', 'c4_Train_18126010.abf', 'c4_Test_18126011.abf'],
         ['c5_AEC_18126014.abf', 'c5_Train_18126015.abf', 'c5_Test_18126016.abf'],
         ['c6_AEC_18126020.abf', 'c6_Train_18126021.abf', 'c6_Test_18126022.abf'],
         ['c7_AEC_18126025.abf', 'c7_Train_18126026.abf', 'c7_Test_18126027.abf'],
         ['c8_AEC_18201000.abf', 'c8_Train_18201001.abf', 'c8_Test_18201002.abf'],
         ['c9_AEC_18201013.abf', 'c9_Train_18201014.abf', 'c9_Test_18201015.abf'],
         ['c10_AEC_18201030.abf', 'c10_Train_18201031.abf', 'c10_Test_18201032.abf'],
         ['c11_AEC_18201035.abf', 'c11_Train_18201036.abf', 'c11_Test_18201037.abf'],
         ['c12_AEC_18309019.abf', 'c12_Train_18309020.abf', 'c12_Test_18309021.abf'],
         ['c13_AEC_18309022.abf', 'c13_Train_18309023.abf', 'c13_Test_18309024.abf']]

experiments = []

print 'LOADING DATA'
for i in range(len(cells)):

    print '\rLoading cell {}'.format(i),

    with gagProcess():

        #Initialize experiment.
        experiment_tmp = Experiment('Cell {}'.format(i), 0.1)

        # Read in file.
        experiment_tmp.setAECTrace('Axon', fname = path + cells[i][0],
                                   V_channel = 0, I_channel = 1)
        experiment_tmp.addTrainingSetTrace('Axon', fname = path + cells[i][1],
                                           V_channel = 0, I_channel = 1)
        experiment_tmp.addTestSetTrace('Axon', fname = path + cells[i][2],
                                       V_channel = 0, I_channel = 1)

    # Store experiment in experiments list.
    experiments.append(experiment_tmp)

print '\nDone!\n'


#%% LOWPASS FILTER V AND I DATA

butter_filter_cutoff = 1000.
butter_filter_order = 3

v_reject_thresh = -80.

print 'FILTERING TRACES & SETTING ROI'
for i in range(len(experiments)):

    print '\rFiltering and selecting for cell {}'.format(i),

    # Filter training data.
    for tr in experiments[i].trainingset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 59000]])

        boolvec = tr.V > v_reject_thresh
        boolvec[:10000] = False

        tr.setROI_Bool(boolvec)


    # Filter test data.
    for tr in experiments[i].testset_traces:
        tr.butterLowpassFilter(butter_filter_cutoff, butter_filter_order)
        tr.setROI([[1000, 9000]])

        boolvec = tr.V > v_reject_thresh
        boolvec[:10000] = False

        tr.setROI_Bool(boolvec)

print '\nDone!\n'


#%% PERFORM AEC

AEC_objs = []

print 'PERFORMING AEC'
for i in range(len(experiments)):

    print '\rCompensating recordings from cell {}'.format(i),

    with gagProcess():

        # Initialize AEC.
        AEC_tmp = AEC_Badel(experiments[i].dt)

        # Define metaparameters.
        AEC_tmp.K_opt.setMetaParameters(length = 500,
                                        binsize_lb = experiments[i].dt,
                                        binsize_ub = 100.,
                                        slope = 5.0,
                                        clamp_period = 0.1)
        AEC_tmp.p_expFitRange = [1., 500.]
        AEC_tmp.p_nbRep = 30

        # Perform AEC.
        experiments[i].setAEC(AEC_tmp)
        experiments[i].performAEC()

    # Save AEC to AEC_objs list.
    AEC_objs.append(AEC_tmp)

print '\nDone!\n'


#%% FIT GIF

Base_GIFs = []
KCond_GIFs = []


plt.figure()
dV_p = plt.subplot(121)
plt.title('Var explained on dV')
plt.ylabel('Var explained (%)')
plt.xticks([0, 1], ['Base model', 'Cond model'])

V_p = plt.subplot(122)
plt.title('Var explained on V')
plt.ylabel('Var explained (%)')
plt.xticks([0, 1], ['Base model', 'Cond model'])


print 'FITTING GIFs'
print '\nCell no.{:>10}{:>10}{:>10}{:>10}'.format('Base dV', 'Base V', 'K dV', 'K V')
print '_________________________________________________________'
for i in range(len(experiments)):

    with gagProcess():

        # Initialize GIF.
        GIF_tmp = SubthreshGIF(experiments[i].dt)

        # Perform fit.
        GIF_tmp.fit(experiments[i])


        # Initialize KGIF.
        KGIF_tmp = SubthreshGIF_K(experiments[i].dt)

        KGIF_tmp = SubthreshGIF_K(0.1)

        # Define parameters
        KGIF_tmp.m_Vhalf = -27
        KGIF_tmp.m_k = 0.113
        KGIF_tmp.m_tau = 1.

        KGIF_tmp.h_Vhalf = -59.9
        KGIF_tmp.h_k = -0.166
        KGIF_tmp.h_tau = 50.

        KGIF_tmp.n_Vhalf = -16.9
        KGIF_tmp.n_k = 0.114
        KGIF_tmp.n_tau = 100.

        KGIF_tmp.E_K = -101.

        # Fit KGIF.
        KGIF_tmp.fit(experiments[i])


    base_vexp_dV    = 100. * np.round(GIF_tmp.var_explained_dV, 3)
    base_vexp_V     = 100. * np.round(GIF_tmp.var_explained_V, 3)
    K_vexp_dV   = 100. * np.round(KGIF_tmp.var_explained_dV, 3)
    K_vexp_V    = 100. * np.round(KGIF_tmp.var_explained_V, 3)

    dV_p.plot([0, 1], [base_vexp_dV, K_vexp_dV], 'k-')
    V_p.plot([0, 1], [base_vexp_V, K_vexp_V], 'k-')

    print '{:>3}{:>10}%{:>10}%{:>10}%{:>10}%'.format(
            i,
            base_vexp_dV,
            base_vexp_V,
            K_vexp_dV,
            K_vexp_V)

    Base_GIFs.append(GIF_tmp)
    KCond_GIFs.append(KGIF_tmp)

dV_p.set_ylim(-5, 105)
dV_p.set_xlim(-0.5, 1.5)
V_p.set_ylim(-5, 105)
V_p.set_xlim(-0.5, 1.5)

plt.tight_layout()
plt.show()

print '\nDone!\n'


#%% PLOT FIT

"""
print 'PLOTTING FIT ON TRAINING SET'
for i in range(len(experiments)):

    Base_GIFs[i].plotFit('Base GIF {}'.format(i))

    KCond_GIFs[i].plotFit('KCond GIF {}'.format(i))

print 'Done!\n'
"""


#%% COMPARE FITS OF BOTH MODELS ON TRAINING SET

for i in range(len(experiments)):

    plt.figure(figsize = (10, 5))

    V_p = plt.subplot(211)
    plt.title('Voltage traces')
    plt.ylabel('V (mV)')
    plt.xlabel('Time (ms)')

    dV_p = plt.subplot(212)
    plt.title('dV traces')
    plt.ylabel('dV/dt (mV/ms)')
    plt.xlabel('Time (ms)')

    t_V = np.arange(0,
                    np.round(len(Base_GIFs[i].V_data[0])*Base_GIFs[i].dt, 1),
                    Base_GIFs[i].dt)
    t_dV = np.arange(0,
                     np.round(len(Base_GIFs[i].dV_data)*Base_GIFs[i].dt, 1),
                     Base_GIFs[i].dt)

    assert len(t_V) == len(Base_GIFs[i].V_data[0]), 'time and V_vectors not of equal lengths'
    assert len(t_dV) == len(Base_GIFs[i].dV_data), 'time and dV_vectors not of equal lengths'

    for j in range(len(Base_GIFs[i].V_data)):

        # Only label the first line.
        if j == 0:
            V_p.plot(t_V, Base_GIFs[i].V_data[j],
                     'k-', linewidth = 0.5, label = 'Real')
            V_p.plot(t_V, Base_GIFs[i].V_sim[j],
                     'r-', linewidth = 0.5, alpha = 0.7, label = 'Linear model')
            V_p.plot(t_V, KCond_GIFs[i].V_sim[j],
                     'b-', linewidth = 0.5, alpha = 0.7, label = 'Linear model + K')

        else:
            V_p.plot(t_V, Base_GIFs[i].V_data[j],
                     'k-', linewidth = 0.5)
            V_p.plot(t_V, Base_GIFs[i].V_sim[j],
                     'r-', linewidth = 0.5)
            V_p.plot(t_V, KCond_GIFs[i].V_sim[j],
                     'b-', linewidth = 0.5, alpha = 0.7)


    dV_p.plot(t_dV, Base_GIFs[i].dV_data, 'k-', label = 'Real')
    dV_p.plot(t_dV, Base_GIFs[i].dV_fitted, 'r-', alpha = 0.7, label = 'Base GIF')
    dV_p.plot(t_dV, KCond_GIFs[i].dV_fitted, 'b-', alpha = 0.7, label = 'KCond GIF')

    V_p.legend()
    dV_p.legend()

    plt.tight_layout()

    plt.suptitle('Cell {}'.format(i))
    plt.subplots_adjust(top = 0.85)

    plt.show()


#%% COMPARE MODEL RESIDUALS

# Bin residuals according to V and populate a pair of arrays with this info
# to use for plotting.

bins = np.linspace(-120, -30, 20)
bin_centres = (bins[1:] + bins[:-1]) / 2.

V_arr_base = np.tile(bin_centres[:, np.newaxis], len(Base_GIFs))
err_arr_base = np.empty_like(V_arr_base, dtype = np.float64)
err_arr_base[:, :] = np.NAN

V_arr_K = np.tile(bin_centres[:, np.newaxis], len(KCond_GIFs))
err_arr_K = np.empty_like(V_arr_K, dtype = np.float64)
err_arr_K[:, :] = np.NAN

del bin_centres


for i in range(len(Base_GIFs)):

    if i in [3, 5]:
        continue

    # Collect residuals of KCond GIF
    V, err = Base_GIFs[i].getResiduals_V(bins)

    assert len(V) == len(err), 'V and err are not the same length!'

    inds = np.digitize(V, bins) - 1
    err_arr_base[inds, i] = err


    # Collect residuals of KCond GIF
    V, err = KCond_GIFs[i].getResiduals_V(bins)

    assert len(V) == len(err), 'V and err are not the same length!'

    inds = np.digitize(V, bins) - 1
    err_arr_K[inds, i] = err


plt.figure(figsize = (4, 3.5))

plt.subplot(111)
plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
plt.plot(V_arr_base, err_arr_base, 'r-', markerfacecolor = 'none', alpha = 0.5)
plt.plot(V_arr_K, err_arr_K, 'b-', markerfacecolor = 'none', alpha = 0.5)
plt.plot(np.nanmean(V_arr_base, axis = 1), np.nanmean(err_arr_base, axis = 1), '-',
         color = 'r', label = 'Linear model')
plt.plot(np.nanmean(V_arr_K, axis = 1), np.nanmean(err_arr_K, axis = 1), '-',
         color = (0.1, 0.1, 0.9), label = 'Linear model + gK')
plt.legend()

plt.xlabel('Vm (mV)')
plt.ylabel('Model error (mV)')

plt.tight_layout()
plt.show()



#%% COMPARE FIT ON TEST SET

residuals_K = []
residuals_base = []

V_test_ls = []

print('\nMEAN RESIDUALS')
print('{1:>13}{0:>10}'.format('K', 'Base'))
print('_________________________')
for i in range(len(experiments)):

    ### Skip crappy cells.
    if i in [10]:
        continue

    ### Get raw residuals

    # Allocate arrays to hold test data.
    I_test = np.zeros((len(experiments[i].testset_traces[0].I),
                       len(experiments[i].testset_traces)),
                      dtype = np.float64)
    V_test = np.zeros_like(I_test, dtype = np.float64)
    V_sim_K = np.zeros_like(I_test, dtype = np.float64)
    V_sim_base = np.zeros_like(I_test, dtype = np.float64)

    for j in range(len(experiments[i].testset_traces)):
        # Get experimental data.
        tr = experiments[i].testset_traces[j]
        I_test[:, j] = tr.I
        V_test[:, j] = tr.V

        # Get simulated data.
        time, V_sim_K_ij, m, h, n = KCond_GIFs[i].simulate(tr.I, tr.V[0])
        V_sim_K[:, j] = V_sim_K_ij

        time, V_sim_base_ij = Base_GIFs[i].simulate(tr.I, tr.V[0])
        V_sim_base[:, j] = V_sim_base_ij

    del (j, tr, time, V_sim_K_ij, m, h, n, V_sim_base_ij)

    # Get residuals.
    residuals_K.append((V_sim_K - V_test)**2)
    residuals_base.append((V_sim_base - V_test)**2)

    # Get voltage along which to bin residuals.
    V_test_ls.append(V_test)

    #print('{0:>3}{2:>10.1f}{1:>10.1f}'.format(
    #        i, residuals_K[i].mean(), residuals_base[i].mean()))



### Bin residuals as a function of V

# Make bins.
bins = np.linspace(-120, -30, 20)
bin_centres = (bins[1:] + bins[:-1]) / 2.

V_arr_base = np.tile(bin_centres[:, np.newaxis], len(residuals_base))
err_arr_base = np.empty_like(V_arr_base, dtype = np.float64)
err_arr_base[:, :] = np.NAN

V_arr_K = np.tile(bin_centres[:, np.newaxis], len(residuals_K))
err_arr_K = np.empty_like(V_arr_K, dtype = np.float64)
err_arr_K[:, :] = np.NAN

del bin_centres

for i in range(len(residuals_K)):

    # Get binned residuals on base model
    err, V, bin_no = stats.binned_statistic(V_test_ls[i].flatten(),
                residuals_base[i].flatten(),
                bins = bins)
    inds = np.digitize(V, bins) - 1
    err_arr_base[:, i] = err

    # Get binned residuals on KCond model
    err, V, bin_no = stats.binned_statistic(V_test_ls[i].flatten(),
                residuals_K[i].flatten(),
                bins = bins)
    inds = np.digitize(V, bins) - 1
    err_arr_K[:, i] = err


### Make figure

plt.figure(figsize = (7, 5))

plt.subplot(111)
plt.axhline(color = 'k', linestyle = 'dashed', linewidth = 0.5)
plt.plot(V_arr_base, err_arr_base, 'r-', markerfacecolor = 'none', alpha = 0.1)
plt.plot(V_arr_K, err_arr_K, 'b-', markerfacecolor = 'none', alpha = 0.1)
plt.plot(np.nanmean(V_arr_base, axis = 1), np.nanmean(err_arr_base, axis = 1), '-',
         color = 'r', label = 'Linear model')
plt.plot(np.nanmean(V_arr_K, axis = 1), np.nanmean(err_arr_K, axis = 1), '-',
         color = (0.1, 0.1, 0.9), label = 'Linear model + gK')

for i in range(V_arr_base.shape[0]):

    if np.isnan(np.nanmean(V_arr_base[i, :])):
        continue

    W, p = stats.wilcoxon(err_arr_base[i, :], err_arr_K[i, :])
    plt.text(V_arr_base[i, 0], -30, str(round(p, 2)),
             horizontalalignment = 'center')

plt.ylim(-40, plt.ylim()[1])

plt.legend()

plt.xlabel('Vm (mV)')
plt.ylabel('Model error (mV^2)')

plt.tight_layout()
plt.show()

#%%
### COMPARE FIT AT SPECIFIC VOLTAGES

cells_to_exclude = []
cells_to_use = [i for i in range(err_arr_base.shape[1]) if i not in cells_to_exclude]
del cells_to_exclude

err_arr_base_stats = np.sqrt(err_arr_base[:, cells_to_use])
err_arr_K_stats = np.sqrt(err_arr_K[:, cells_to_use])

print "\n"
print stats.ttest_rel(err_arr_base_stats[14, :], err_arr_K_stats[14, :], nan_policy = 'omit')
print stats.ttest_rel(err_arr_base_stats[15, :], err_arr_K_stats[15, :], nan_policy = 'omit')
print stats.ttest_rel(err_arr_base_stats[16, :], err_arr_K_stats[16, :], nan_policy = 'omit')


print '\n'
print stats.wilcoxon(err_arr_base_stats[14, :], err_arr_K_stats[14, :])
print stats.wilcoxon(err_arr_base_stats[15, :], err_arr_K_stats[15, :])
print stats.wilcoxon(err_arr_base_stats[16, :], err_arr_K_stats[16, :])


#%% PLOT GBAR ESTIMATES

gk_leak_pdata = []
gbase_leak_pdata = []
gk1_pdata = []
gk2_pdata = []

print 'PLOTTING GBAR ESTIMATES'
for KGIF in KCond_GIFs:

    gk_leak_pdata.append(KGIF.gl)
    gk1_pdata.append(KGIF.gbar_K1)
    gk2_pdata.append(KGIF.gbar_K2)

for GIF in Base_GIFs:

    gbase_leak_pdata.append(GIF.gl)


plt.figure()

plt.subplot(111)
plt.title('Estimated maximal conductances')
plt.plot([0] * len(gk_leak_pdata),
         gk_leak_pdata,
         'ko', markersize = 20, markerfacecolor = 'gray',
         markeredgecolor = 'k', alpha = 0.5)
plt.plot([1] * len(gk1_pdata),
         gk1_pdata,
         'ko', markersize = 20, markerfacecolor = 'gray',
         markeredgecolor = 'k', alpha = 0.5)
plt.plot([2] * len(gk2_pdata),
         gk2_pdata,
         'ko', markersize = 20, markerfacecolor = 'gray',
         markeredgecolor = 'k', alpha = 0.5)

plt.ylabel('gbar')
plt.xticks([0, 1, 2], ['Leak conductance', 'Conductance 1', 'Conductance 2'], rotation = 45)
plt.xlim(-0.5, 2.5)

plt.tight_layout()
plt.show()



plt.figure(figsize = (3, 4))

plt.subplot(111)
plt.errorbar([0],
         np.mean(gk_leak_pdata),
         np.std(gk_leak_pdata)/np.sqrt(len(gk_leak_pdata)),
         fmt = 'ko', markersize = 10)
plt.errorbar([1],
         np.mean(gk1_pdata),
         np.std(gk1_pdata)/np.sqrt(len(gk1_pdata)),
         fmt = 'ko', markersize = 10)
plt.errorbar([2],
         np.mean(gk2_pdata),
         np.std(gk2_pdata)/np.sqrt(len(gk2_pdata)),
         fmt = 'ko', markersize = 10)

plt.ylabel('Maximal conductance (nS)')
plt.xlabel('Conductance')
plt.xticks([0, 1, 2], ['gl', 'gk1', 'gk2'])
plt.xlim(-0.5, 2.5)
plt.ylim(0, 0.015)

plt.tight_layout()
plt.show()


plt.figure()

plt.subplot(211)
plt.title('Estimated leak conductances')
plt.hist(gbase_leak_pdata, color = 'k', alpha = 0.5,
         label = 'Linear model')
plt.hist(gk_leak_pdata, color = 'b', alpha = 0.5,
         label = 'Linear model + gk')
plt.ylabel('Count')
plt.xlabel('gl (nS)')
plt.legend()

plt.subplot(212)
plt.title('Estimated active conductances')
plt.hist(gk1_pdata, color = (0.9, 0.1, 0.1), alpha = 0.5,
         label = 'gk1')
plt.hist(gk2_pdata, color = (0.1, 0.9, 0.1), alpha = 0.5,
         label = 'gk2')
plt.ylabel('Count')
plt.xlabel('g (nS)')
plt.legend()

plt.tight_layout()
plt.show()


#%% MAKE BEESWARM PLOT OF ALL PARAMETER ESTIMATES

# Place parameter estimates into a DataFrame

param_dict = {
        'Model': [],
        'Parameter': [],
        'Value': [],
        }

for i in range(len(Base_GIFs)):


    # Get parameter estimates for base model
    params_tmp = ['R', 'C', 'tau']
    vals_tmp = [1./Base_GIFs[i].gl, Base_GIFs[i].C,
                1./Base_GIFs[i].gl * Base_GIFs[i].C]

    assert len(params_tmp) == len(vals_tmp), 'param labels and vals not same length'

    mod_tmp = ['Base'] * len(params_tmp)

    param_dict['Model'].extend(mod_tmp)
    param_dict['Parameter'].extend(params_tmp)
    param_dict['Value'].extend(vals_tmp)


    # Get parameter estimates for active model
    params_tmp = ['R', 'C', 'tau', 'gl', 'gk1', 'gk2']
    vals_tmp = [1./KCond_GIFs[i].gl, KCond_GIFs[i].C,
                1./KCond_GIFs[i].gl * KCond_GIFs[i].C,
                KCond_GIFs[i].gl, KCond_GIFs[i].gbar_K1,
                KCond_GIFs[i].gbar_K2]

    assert len(params_tmp) == len(vals_tmp)

    mod_tmp = ['KCond'] * len(params_tmp)

    param_dict['Model'].extend(mod_tmp)
    param_dict['Parameter'].extend(params_tmp)
    param_dict['Value'].extend(vals_tmp)


# Put param dict into dataframe for plotting using seaborn & clean up
param_df = pd.DataFrame(data = param_dict)
del param_dict, params_tmp, vals_tmp, mod_tmp


# Make figure

plt.figure(figsize = (5.5, 3.5))

R_plot = plt.subplot2grid((2, 3), (0, 0))
df_tmp = param_df.loc[param_df['Parameter'] == 'R', :]
sns.swarmplot(x = df_tmp['Model'], y = df_tmp['Value']/1e3, ax = R_plot)
R_plot.set_ylabel('R (GOhm)')
R_plot.set_ylim(0, R_plot.get_ylim()[1])

C_plot = plt.subplot2grid((2, 3), (0, 1))
df_tmp = param_df.loc[param_df['Parameter'] == 'C', :]
sns.swarmplot(x = df_tmp['Model'], y = df_tmp['Value'] * 1e3, ax = C_plot)
C_plot.set_ylabel('C (pF)')

tau_plot = plt.subplot2grid((2, 3), (0, 2))
df_tmp = param_df.loc[param_df['Parameter'] == 'tau', :]
sns.swarmplot(x = df_tmp['Model'], y = df_tmp['Value'], ax = tau_plot)
tau_plot.set_ylabel('tau (ms)')
tau_plot.set_ylim(0, tau_plot.get_ylim()[1])

g_plot = plt.subplot2grid((2, 3), (1, 0), colspan = 2)
param_checker = np.vectorize(lambda x: x in ['gl', 'gk1', 'gk2'])
selection = np.logical_and(param_df['Model'] == 'KCond',
                           param_checker(param_df['Parameter']))
df_tmp = param_df.loc[selection, :]
sns.swarmplot(x = df_tmp['Parameter'], y = df_tmp['Value'], color = 'k', ax = g_plot)
g_plot.set_ylabel('g (nS)')

corr_plot = plt.subplot2grid((2, 3), (1, 2))
param_checker = np.vectorize(lambda x: x in ['C', 'tau'])
selection = np.logical_and(param_df['Model'] == 'Base',
                           param_checker(param_df['Parameter']))
df_tmp = param_df.loc[selection, :]
corr_plot.plot(df_tmp.loc[df_tmp['Parameter'] == 'tau', 'Value'],
               df_tmp.loc[df_tmp['Parameter'] == 'C', 'Value'] * 1e3,
               'ko',
               alpha = 0.5)
corr_plot.set_ylabel('C (pF)')
corr_plot.set_xlabel('tau (ms)')
corr_plot.set_ylim(0, corr_plot.get_ylim()[1])
corr_plot.set_xlim(0, corr_plot.get_xlim()[1])

plt.tight_layout()


#%% SHOW SIMULATED V-CLAMP

for i in range(len(KCond_GIFs)):

    KGIF = KCond_GIFs[i]

    plt.figure(figsize = (6, 4))
    plt.subplot(111)
    plt.title('Simulated voltage clamp test - {}'.format(i))
    plt.ylabel('Holding current (nA)')
    plt.xlabel('Time (ms)')

    for V in np.arange(-60, -20, 10):

        I_vec = KGIF.simulateVClamp(500, V, -90, True)[1]
        t_vec = np.arange(0, int(np.round(len(I_vec) * KGIF.dt)), KGIF.dt)

        plt.plot(t_vec, I_vec, label = str(V) + 'mV')

    plt.legend()

    plt.tight_layout()


#%% PLOT POWER SPECTRUM DENSITY

print 'EXTRACTING/PLOTTING POWER SPECTRUM DENSITY'
for i in range(len(experiments)):

    print '\rExtracting cell {}'.format(i),

    Base_GIFs[i].plotPowerSpectrumDensity('Base GIF {}'.format(i))

    KCond_GIFs[i].plotPowerSpectrumDensity('KCond GIF {}'.format(i))


#%% EXTRACT DETAILED SIMULATED POWER SPECTRUM Density

"""
Get the power spectrum density of model voltage response to very long noisy
input.
"""

noise_length = int(1e6)
noise_sigma = 0.005                             # (nA)
noise_offsets = np.linspace(-0.010, 0.030, 3)  # (nA)
truncate = int(5e4)

very_long_noise = {
    'I': [],
    'V': [],
    '_dt': [],
    'I_f': [],
    'I_PSD': [],
    'V_f': [],
    'V_PSD': []
}

# Perform simulations and extract PSD.
for i in range(len(KCond_GIFs)):

    print 'Getting frequency response for cell {}'.format(i)

    I_arr = np.empty((noise_length, len(noise_offsets)),
                     dtype = np.float64)
    V_arr = np.empty_like(I_arr)

    base_noise = Tools.generateOUprocess(noise_length * 0.1, 5., 0.,
                                              noise_sigma, 0.1)

    # Offset noise.
    for j in range(I_arr.shape[1]):
        I_arr[:, j] = base_noise + noise_offsets[j]

    # Add offset that was in experimental data to keep cell at ~-60mV.
    I_arr += experiments[i].trainingset_traces[0].I[:100].mean()

    V0 = experiments[i].trainingset_traces[0].V[:100].mean()

    # Simulate V response to noise.
    for j in range(I_arr.shape[1]):
        print '\rSimulating voltage response: {:0.1f}%'.format(100. * (j+1)/I_arr.shape[1]),
        _, V_arr[:, j], _, _, _ = KCond_GIFs[i].simulate(I_arr[:, j], V0)

    # Truncate V/I arrays to remove slow drift to equilibrium voltage.
    I_arr = I_arr[truncate:, :]
    V_arr = V_arr[truncate:, :]

    # Add output to master dict.
    very_long_noise['I'].append(I_arr)
    very_long_noise['V'].append(V_arr)
    very_long_noise['_dt'].append(KCond_GIFs[i].dt)

    print '\nDone cell {}!'.format(i)



#%% Extract PSD

def PSDworker(args):

    """
    Worker function to extract PSD of V and I arrays.
    Used for parallelization with multiprocessing.

    Args should be a tuple of V_arr, I_arr, and dt.
    """

    V_arr = args[0]
    I_arr = args[1]
    dt = args[2]

    I_f_arr = []
    I_PSD_arr = []
    V_f_arr = []
    V_PSD_arr = []

    # Get PSD
    for j in range(V_arr.shape[1]):
        tr = Trace(V_arr[:, j], I_arr[:, j], V_arr.shape[0] * dt, dt)
        V_f, V_PSD, I_f, I_PSD = tr.extractPowerSpectrumDensity()

        I_f_arr.append(I_f)
        I_PSD_arr.append(I_PSD)
        V_f_arr.append(V_f)
        V_PSD_arr.append(V_PSD)

    # Convert PSD lists to arrays.
    I_f_arr = np.array(I_f_arr).T
    I_PSD_arr = np.array(I_PSD_arr).T
    V_f_arr = np.array(V_f_arr).T
    V_PSD_arr = np.array(V_PSD_arr).T

    return (I_f_arr, I_PSD_arr, V_f_arr, V_PSD_arr)

# Creat an iterable to pass to the worker process.
noise_input_iter = itertools.izip(very_long_noise['V'], very_long_noise['I'], very_long_noise['_dt'])

# Parallelize.
if __name__ == '__main__':

    print 'Extracting PSDs in parallel.'

    pool_ = mp.Pool()

    for out in pool_.imap(PSDworker, noise_input_iter):

        very_long_noise['I_f'].append(out[0])
        very_long_noise['I_PSD'].append(out[1])
        very_long_noise['V_f'].append(out[2])
        very_long_noise['V_PSD'].append(out[3])

    pool_.close()
    pool_.join()

    print 'Done!'

else:
    print 'Cannot execute parallized PSD extraction outside of __main__.'
    print 'Iterating normally instead.'

    for out in itertools.imap(PSDworker, noise_input_iter):
        very_long_noise['I_f'].append(out[0])
        very_long_noise['I_PSD'].append(out[1])
        very_long_noise['V_f'].append(out[2])
        very_long_noise['V_PSD'].append(out[3])


#%% PLOT DETAILED SIMULATED PSD

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
plt.plot(very_long_noise['V_f'][0],
         very_long_noise['V_PSD'][0])

plt.figure()
ax = plt.subplot(111)
plt.plot(very_long_noise['V'][0])
