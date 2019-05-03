#%% IMPORT MODULES

from __future__ import division

import pickle
import os

from src.GIF import GIF
from src.Filter_Exps import Filter_Exps
from src.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = os.path.join('data', 'processed', '5HT_fastnoise')

with open(os.path.join(DATA_PATH, '5HT_goodcells.ldat'), 'rb') as f:
    experiments = pickle.load(f)
    f.close()


#%% FIT GIFS

MODEL_PATH = os.path.join('data', 'models', '5HT')

GIFs = []

for i, expt in enumerate(experiments):

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = GIF(0.1)

    with gagProcess():

        # Define parameters
        tmp_GIF.Tref = 6.5

        tmp_GIF.eta = Filter_Exps()
        tmp_GIF.eta.setFilter_Timescales([3, 10, 30, 100, 300, 1000, 3000])

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

with open(os.path.join(MODEL_PATH, 'sergifs.lmod'), 'wb') as f:
    pickle.dump(GIFs, f)

