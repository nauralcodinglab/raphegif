#%% IMPORT MODULES

from __future__ import division

import os
import pickle


#%% COPY MODEL

# Load model.
MOD_PATH = os.path.join('data', 'models', 'GIF_network')
with open(os.path.join(MOD_PATH, 'subsample.mod'), 'rb') as f:
    mod = pickle.load(f)
    f.close()

# Do nothing.

# Save model.
with open(os.path.join(MOD_PATH, 'subsample_base.mod'), 'wb') as f:
    pickle.dump(mod, f)
    f.close()


