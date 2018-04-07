#%% IMPORT MODULES

from __future__ import division
import os
import sys
import pickle

import numpy as np

sys.path.append('analysis/subthresh_mod_selection')
from ModMats import ModMats


#%% LOAD DATA

PICKLE_PATH = 'data/subthreshold_expts/compensated_recs/'

print 'LOADING DATA'
model_matrices = []

fnames = [fname for fname in os.listdir(PICKLE_PATH) if fname[-4:].lower() == '.pyc']
for fname in fnames:

    with open(PICKLE_PATH + fname, 'rb') as f:

        modmat_tmp = ModMats(0.1)
        modmat_tmp = pickle.load(f)

    model_matrices.append(modmat_tmp)

print 'Done!'
