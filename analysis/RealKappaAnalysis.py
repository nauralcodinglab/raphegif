#%% IMPORT MODULES

from __future__ import division
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal

import sys
sys.path.append('src')
sys.path.append('analysis/gating')

from SubthreshGIF_K import SubthreshGIF_K
import Tools
from cell_class import Cell

%matplotlib qt5


#%% READ IN FILES

"""
Load recordings using the read_ABF method from my custom Cell class (depends on neo).
Each recording is one sweep of recorded voltage noise in response to noisy current injection.
"""

PATH = 'data/membrane_noise/'
fnames = [fname for fname in os.listdir(PATH) if fname[-4:].lower() == '.abf']
fnames.sort()
recs = [Cell.read_ABF(Cell(), PATH + fname)[0] for fname in fnames]


#%% GET MEMBRANE FILTER

"""
Extract the membrane filter based on the power spectra of injected current and
recorded voltage.
"""

V_PSDs = [signal.welch(rec[0, :, :].flatten(), 1e4, 'hann', 3e4) for rec in recs]
I_PSDs = [signal.welch(rec[1, :, :].flatten(), 1e4, 'hann', 3e4) for rec in recs]
kappas = np.array([np.sqrt(V_PSD[1] / I_PSD[1]) for V_PSD, I_PSD in zip(V_PSDs, I_PSDs)]) * 1e3
fs = np.array([V_PSD[0] for V_PSD in V_PSDs])
mean_Vs = [rec[0, :, :].mean() for rec in recs]

kappas = kappas[:9]
fs = fs[:9]
mean_Vs = np.broadcast_to(np.array(mean_Vs)[:9, np.newaxis], kappas.shape)


#%% DISPLAY MEMBRANE FILTER

"""
Use matplotlib to construct a 3D graph that shows the voltage-dependence of the
membrane filter.
"""

# Create temporary arrays to use for 3D plot.
F = fs.T
V = mean_Vs.T
I = kappas.T

# Subset arrays based on frequency.

freq_sub = np.min(np.where(F[:, 0] >= 1e2)[0])
F = F[1:freq_sub, :]
V = V[1:freq_sub, :]
I = I[1:freq_sub, :]

F = np.log10(F)

# Make figure.
fig3d = plt.figure(figsize = (5, 5))

ax0 = fig3d.add_subplot(111, projection = '3d')
ax0.set_title('Membrane filter from unidentified neuron')
ax0.plot_surface(F, V, I, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
#ax0.set_ylim3d(ax0.get_ylim3d()[1], ax0.get_ylim3d()[0])
#ax0.set_xticklabels([r'$\displaystyle10^{}$!'.format(tick) for tick in ax0.get_xticks()])
ax0.set_xlabel('log10(f/f0)')
ax0.set_ylabel('Vm (mV)')
ax0.set_zlabel('Impedance (MOhm)')

plt.show()
