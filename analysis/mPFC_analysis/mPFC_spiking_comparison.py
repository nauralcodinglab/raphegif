#%% IMPORT MODULES

import sys
sys.path.append('./src')

import matplotlib.pyplot as plt
import pandas as pd

from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *


#%% IMPORT FILES

FNAMES_PATH = './data/raw/mPFC/fnames.csv'
DATA_PATH = './data/raw/mPFC/mPFC_spiking/'
dt = 0.1

make_plots = True

fnames = pd.read_csv(FNAMES_PATH)

mPFC_experiments = []

for i in range(fnames.shape[0]):

    if fnames.loc[i, 'TTX'] == 0:

        tmp_experiment = Experiment(fnames.loc[i, 'Experimenter'] + fnames.loc[i, 'Cell_ID'], dt)
        tmp_experiment.setAECTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'AEC'],
            V_channel = 0, I_channel = 1)
        tmp_experiment.addTrainingSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Train'],
            V_channel = 0, I_channel = 1)
        tmp_experiment.addTestSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Test'],
            V_channel = 0, I_channel = 1)

        if make_plots:
            tmp_experiment.plotTrainingSet()

        mPFC_experiments.append(tmp_experiment)

    else:
        continue


#%% PERFORM AEC

for expt in mPFC_experiments:

    tmp_AEC = AEC_Badel(expt.dt)

    # Define metaparametres
    tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
    tmp_AEC.p_expFitRange = [3.0,150.0]
    tmp_AEC.p_nbRep = 15

    # Assign tmp_AEC to myExp and compensate the voltage recordings
    expt.setAEC(tmp_AEC)
    expt.performAEC()


#%% FIT GIF

mPFC_GIFs = []

for expt in mPFC_experiments:

    tmp_GIF = GIF(0.1)

    # Define parameters
    tmp_GIF.Tref = 4.0

    tmp_GIF.eta = Filter_Rect_LogSpaced()
    tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


    tmp_GIF.gamma = Filter_Rect_LogSpaced()
    tmp_GIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

    # Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
    expt.trainingset_traces[0].setROI([[1000,59000]])

    # Perform the fit
    tmp_GIF.fit(expt, DT_beforeSpike=5.0)

    mPFC_GIFs.append(tmp_GIF)


#%% EVALUATE FIT

make_plots = True

mPFC_Md_vals = []
mPFC_predictions = []

for expt, GIF in zip(mPFC_experiments, mPFC_GIFs):

    tmp_prediction = expt.predictSpikes(GIF, nb_rep = 500)
    tmp_Md = tmp_prediction.computeMD_Kistler(4.0, 0.1)

    mPFC_Md_vals.append(tmp_Md)
    mPFC_predictions.append(tmp_prediction)

    if make_plots:
        tmp_prediction.plotRaster(delta = 1000.)

#%%

mPFC_var_explained_V_ls = [mod.var_explained_V for mod in mPFC_GIFs]
mPFC_var_explained_dV_ls = [mod.var_explained_dV for mod in mPFC_GIFs]

plt.figure()

plt.subplot(131)
plt.title('Training R^2 on dV')
plt.plot([0 for i in range(len(mPFC_var_explained_dV_ls))], mPFC_var_explained_dV_ls,
    'ko', markersize = 10, alpha = 0.3)
plt.ylabel('R^2')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.subplot(132)
plt.title('Training R^2 on V')
plt.plot([0 for i in range(len(mPFC_var_explained_V_ls))], mPFC_var_explained_V_ls,
    'ko', markersize = 10, alpha = 0.3)
plt.ylabel('R^2')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.subplot(133)
plt.title('Md* 4ms')
plt.plot([0 for i in range(len(mPFC_Md_vals))], mPFC_Md_vals,
    'ko', markersize = 10, alpha = 0.3)
plt.ylabel('Md*')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.tight_layout()
plt.show()
