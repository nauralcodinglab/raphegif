#%% IMPORT MODULES

import sys
sys.path.append('./src')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Rect import Filter_Rect_LinSpaced


#%% IMPORT FILES

FNAMES_PATH = './data/raw/mPFC/fnames.csv'
DATA_PATH = './data/raw/mPFC/mPFC_spiking/'
dt = 0.1

make_plots = True

fnames = pd.read_csv(FNAMES_PATH)

experiments = []

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

        experiments.append(tmp_experiment)

    else:
        continue


#%% PERFORM AEC

for expt in experiments:

    tmp_AEC = AEC_Badel(expt.dt)

    # Define metaparametres
    tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
    tmp_AEC.p_expFitRange = [3.0,150.0]
    tmp_AEC.p_nbRep = 15

    # Assign tmp_AEC to myExp and compensate the voltage recordings
    expt.setAEC(tmp_AEC)
    expt.performAEC()


#%% FIT GIF

GIFs = []

for expt in experiments:

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

    GIFs.append(tmp_GIF)


#%% EVALUATE FIT

make_plots = True

Md_vals = []
predictions = []

for expt, GIF in zip(experiments, GIFs):

    tmp_prediction = expt.predictSpikes(GIF, nb_rep = 500)
    tmp_Md = tmp_prediction.computeMD_Kistler(4.0, 0.1)

    Md_vals.append(tmp_Md)
    predictions.append(tmp_prediction)

    if make_plots:
        tmp_prediction.plotRaster(delta = 1000.)

#%%

Ra_vals = [expt.getAEC().K_e.computeIntegral(expt.dt) for expt in experiments]

var_explained_V_ls = [mod.var_explained_V for mod in GIFs]
var_explained_dV_ls = [mod.var_explained_dV for mod in GIFs]

spec = plt.GridSpec(1, 3)
plt.rc('text', usetex = True)

plt.figure()

plt.suptitle('\\textbf{{Pyramidal neurons}}')

plt.subplot(spec[0, 0])
plt.title('Training $R^2$ on $dV$')
sns.swarmplot(
    [0 for i in range(len(var_explained_dV_ls))],
    var_explained_dV_ls,
    size = 20, color = 'gray', alpha = 0.8
)
plt.ylabel('$R^2$')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.subplot(spec[0, 1])
plt.title('Training $R^2$ on $V$')
sns.swarmplot(
    [0 for i in range(len(var_explained_V_ls))],
    var_explained_V_ls,
    size = 20, color = 'gray', alpha = 0.8
)
plt.ylabel('$R^2$')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.subplot(spec[0, 2])
plt.title('Md* 4ms')
sns.swarmplot(
    [0 for i in range(len(Md_vals))],
    Md_vals,
    size = 20, color = 'gray', alpha = 0.8
)
plt.ylabel('Md*')
plt.ylim(-0.05, 1.05)
plt.xticks([])

plt.subplots_adjust(wspace = 0.7, top = 0.82, bottom = 0.05, right = 0.95, left = 0.05)

plt.savefig('./analysis/mPFC_analysis/pyramidal_neuron_summary.png', dpi = 300)

plt.show()
