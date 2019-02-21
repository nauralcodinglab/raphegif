#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('./analysis/feedforward_gain_modulation/')
import FeedForwardDRN as ff


#%% CREATE INPUT

dt = 0.001

syn_kernels = []

for tau_ in [0.5, 1, 1.5]:
    syn_kernels.append(
        ff.SynapticKernel(
            'alpha', dt = dt, ampli = 1, tau = tau_, kernel_len = tau_ * 4
        )
    )


#%% CREATE DRN MODEL

no_neurons = 300

np.random.seed(45)
random_ser_taus = np.random.gamma(4.52, 0.19, no_neurons) + 0.22
random_gaba_taus = np.random.gamma(4.52, 0.19, no_neurons) + 0.22

DRN_mod = ff.FeedForwardDRN('DRN base', dt)
DRN_mod.construct_ser_mod(10, 1, 1, random_ser_taus)
DRN_mod.construct_gaba_mod(0, 1, 1, random_gaba_taus)
DRN_mod.construct_gaba_kernel(0.01, 0.5, -8, 2.5)
DRN_mod.set_propagation_delay(0.1)


#%%

psth_window_width = 0.1

gain_mods = []

ser_psth_amplis = pd.DataFrame(columns = ['tau', 'ampli', 'ff', 'PSTH_max'])

for kernel in syn_kernels:
    for ampli in np.linspace(20, 50, 10):
        for do_ff in [True, False]:

            tmp_ffmod = DRN_mod.copy('{} -- tau={}, ampli={}, ff={}'.format(
                kernel.kernel_type, kernel.kernel_params['tau'], ampli, do_ff
            ))

            tmp_ffmod.set_ser_Vin(ampli * kernel.tile(no_neurons, pad_length = 1))
            tmp_ffmod.set_gaba_Vin(ampli * kernel.tile(no_neurons, pad_length = 1))

            tmp_ffmod.simulate(feed_forward = do_ff)

            #tmp_ffmod.plot_traces()

            ser_psth_amplis = ser_psth_amplis.append({
                'tau': kernel.kernel_params['tau'],
                'ampli': ampli,
                'ff': do_ff,
                'PSTH_max': tmp_ffmod.ser_psth(psth_window_width).max()
            }, ignore_index = True)
            gain_mods.append(tmp_ffmod)


#%% CREATE FIGURES

for tau in ser_psth_amplis['tau'].unique():

    plt.figure()

    plt.subplot(111)
    plt.title('Input tau={}'.format(tau))

    for ff_state in ser_psth_amplis['ff'].unique():
        tmp = ser_psth_amplis.loc[np.logical_and(ser_psth_amplis['tau'] == tau, ser_psth_amplis['ff'] == ff_state), ['ampli', 'PSTH_max']]
        plt.plot(tmp['ampli'], tmp['PSTH_max'], label = 'Feed-forward {}'.format(ff_state))

    plt.ylabel('PSTH max')
    plt.xlabel('Input amplitude')
    plt.legend()
    plt.show()
