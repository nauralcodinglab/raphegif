#%% IMPORT MODULES

from __future__ import division


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./analysis/feedforward_gain_modulation/')
import FeedForwardDRN as ff


#%% SYNPATIC KERNEL TESTS

# Generate kernels inline.
ff.SynapticKernel(dt = 0.001).generate_alpha(1, 5, 10)
ff.SynapticKernel(dt = 0.001).generate_biexp(0.1, 1, 5, 10)

# Alpha kernel.
alpha_kernel = ff.SynapticKernel('alpha', tau = 0.5, ampli = 30, kernel_len = 5)
alpha_kernel.tile(300, centered = True)
alpha_kernel.tile(300, pad_length = 1)
alpha_kernel.plot(centered = True)
alpha_kernel.kernel_params

# Biexp kernel.
biexp_kernel = ff.SynapticKernel(
    'biexp', tau_rise = 0.05, tau_decay = 0.5, ampli = 3, kernel_len = 5
)
biexp_kernel.tile(300, centered = True)
biexp_kernel.plot(centered = True)
biexp_kernel.kernel_params


#%% CONSTRUCT INPUT

dt = 0.001

no_neurons = 300
sim_len = 3.5

### Step input.

Vin_step = np.zeros((int(sim_len / dt), no_neurons))
Vin_step[int(1/dt):, :] = 23

#plt.plot(Vin_step[:, 0], 'k-')

### Alpha input.
Vin_alpha = alpha_kernel.tile(no_neurons)


#%% PERFORM TEST SIMULATIONS

try:
    DRN_mod = ff.FeedForwardDRN('test', dt)
    DRN_mod.construct_ser_mod(10, 1, 1, 1)
    DRN_mod.construct_gaba_mod(0, 1, 1, 1)
    DRN_mod.construct_gaba_kernel(0.01, 0.5, -8, 2.5)
    #DRN_mod.plot_gaba_kernel()
    DRN_mod.set_propagation_delay(0.1)
except:
    print('Could not instantiate FeedForwardDRN model.')
    raise


# Make a copy to pass step input.
step_mod = DRN_mod.copy('Step input simulation')
step_mod.set_ser_Vin(Vin_step)
step_mod.set_gaba_Vin(Vin_step)

try:
    step_mod.simulate()
except:
    print('FeedForwardDRN.simulate test failed.')
    raise

try:
    step_mod.plot_rasters()
    step_mod.plot_traces()
except:
    print('Could not plot FeedForwardDRN simulation results.')
    raise


# Make a copy to pass alpha input
alpha_mod = DRN_mod.copy('Alpha input simulation')
alpha_mod.set_ser_Vin(alpha_kernel.tile(no_neurons, pad_length = 1))
alpha_mod.set_gaba_Vin(alpha_kernel.tile(no_neurons, pad_length = 1))

try:
    alpha_mod.simulate()
except:
    print('Simulation using alpha input kernel failed.')
    raise

alpha_mod.plot_traces()
