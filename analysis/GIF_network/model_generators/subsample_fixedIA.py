#%% IMPORT MODULES

from __future__ import division

import os
import pickle


#%% MODIFY MODEL

# Load model.
MOD_PATH = os.path.join('data', 'models', 'GIF_network')
with open(os.path.join(MOD_PATH, 'subsample.mod'), 'rb') as f:
    mod = pickle.load(f)
    f.close()

# Fix IA.
mod.name = 'Subsample no IA'
for i in range(len(mod.ser_mod)):
    mod.ser_mod[i].gbar_K1 = 0.010
    mod.ser_mod[i].h_tau = 40.

# Save model.
with open(os.path.join(MOD_PATH, 'subsample_fixedIA.mod'), 'wb') as f:
    pickle.dump(mod, f)
    f.close()


