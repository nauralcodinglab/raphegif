#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np

from src.Tools import generateOUprocess


#%% GENERATE NOISE TO EXPORT

print 'Generating noise.'
distal_input = {
    'ser_input': np.array([generateOUprocess(60000., 100., 0.050, 0.050, 0.1, 42).astype(np.float32)] * 1200, dtype = np.float32),
    'gaba_input': np.array([generateOUprocess(60000, 30., 0.050, 0.050, 0.1, 43).astype(np.float32)] * 800, dtype = np.float32)
}

#%% SAVE NOISE

print 'Saving noise.'
with open(os.path.join('data', 'simulations', 'GIF_network', 'subsample_input.ldat'), 'wb') as f:
    pickle.dump(distal_input, f)
    f.close()
print 'Done!'

