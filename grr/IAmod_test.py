# IMPORT MODULES

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from IAmod import IAmod, Simulation

# TEST

no_neurons = 100
pulse_ampli = 30
baseline_voltage = -60
tau_h_prime = 2.5

high_IA = IAmod(15, tau_h_prime, 2)
high_IA.El = -70
Vin_hi = np.empty((5000, no_neurons), dtype=np.float64)
Vin_hi[:1000, :] = high_IA.ss_clamp(baseline_voltage)
Vin_hi[1000:, :] = high_IA.ss_clamp(baseline_voltage + pulse_ampli)

low_IA = IAmod(1, tau_h_prime, 2)
low_IA.El = -70
Vin_lo = np.empty((5000, no_neurons), dtype=np.float64)
Vin_lo[:1000, :] = low_IA.ss_clamp(baseline_voltage)
Vin_lo[1000:, :] = low_IA.ss_clamp(baseline_voltage + pulse_ampli)

hi_IA_sim = Simulation(high_IA, -60, Vin_hi)
lo_IA_sim = Simulation(low_IA, -60, Vin_lo)

plt.rcdefaults()
plt.rc('text', usetex=True)

hi_IA_sim.simple_plot()
plt.suptitle('$\\bar{{g}}_a^\prime = 10$, $\\tau_h^\prime = 1.5$')
plt.subplots_adjust(top=0.85)
plt.savefig('./figs/ims/hi_IA.png', dpi=300)
plt.show()

lo_IA_sim.simple_plot()
plt.suptitle('$\\bar{{g}}_a^\prime = 1$, $\\tau_h^\prime = 1.5$')
plt.subplots_adjust(top=0.85)
plt.savefig('./figs/ims/lo_IA.png', dpi=300)
plt.show()
