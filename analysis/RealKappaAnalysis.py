#%% IMPORT MODULES

from __future__ import division
import os

import numpy as np
import plotly.offline as pyoff
import plotly.graph_objs as go
from scipy import signal

from src.SubthreshGIF_K import SubthreshGIF_K
from src import Tools
from src.cell_class import Cell


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
Use plotly to construct a 3D graph that shows the voltage-dependence of the
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

# Make exportable interactive 3D plot using plotly.
data = [go.Surface(x = F, y = V, z = I)]
layout = go.Layout(title = 'Membrane filter from unidentified neuron',
scene = go.Scene(
xaxis = {'title': 'log10(f/f0)'},
yaxis = {'title': 'Vm (mV)'},
zaxis = {'title': 'Impedance (MOhm)'}),
autosize = False, width = 500, height = 500,
margin = {'l': 30, 'r': 30, 'b': 30, 't': 60})
pyfig = go.Figure(data = data, layout = layout)
pyoff.plot(pyfig, image = 'png')
