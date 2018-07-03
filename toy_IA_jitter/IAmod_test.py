#%% IMPORT MODULES

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('./toy_IA_jitter')
from IAmod import IAmod, Simulation

#%% TEST

no_neurons = 100
pulse_ampli = 30
tau_h_prime = 2.5

V_baseline = np.zeros((1000, no_neurons))
V_pulse = np.ones((4000, no_neurons)) * pulse_ampli
Vin = np.concatenate((V_baseline, V_pulse), axis = 0)

high_IA = IAmod(15, tau_h_prime, 2)
low_IA = IAmod(1, tau_h_prime, 2)

hi_IA_sim = Simulation(high_IA, -60, Vin)
lo_IA_sim = Simulation(low_IA, -60, Vin)

plt.rcdefaults()
plt.rc('text', usetex = True)

hi_IA_sim.simple_plot()
plt.suptitle('$\\bar{{g}}_a^\prime = 10$, $\\tau_h^\prime = 1.5$')
plt.subplots_adjust(top = 0.85)
plt.savefig('./figs/ims/hi_IA.png', dpi = 300)
plt.show()

lo_IA_sim.simple_plot()
plt.suptitle('$\\bar{{g}}_a^\prime = 1$, $\\tau_h^\prime = 1.5$')
plt.subplots_adjust(top = 0.85)
plt.savefig('./figs/ims/lo_IA.png', dpi = 300)
plt.show()
