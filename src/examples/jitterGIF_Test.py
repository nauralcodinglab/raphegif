#%% IMPORT MODULES

from __future__ import division

import matplotlib.pyplot as plt

import sys
sys.path.append('./src/')
from jitterGIF import jitterGIF


#%% INITIALIZE MODEL PARAMETERS

myGIF = jitterGIF(0.1)

myGIF.El = -60
myGIF.gl = 0.001
myGIF.C = 0.070

myGIF.gbar_K1 = 0
myGIF.gbar_K2 = 0

myGIF.Vthresh = -45
myGIF.Vreset = -75

myGIF.m_Vhalf = -23.7
myGIF.m_k = 0.10
myGIF.m_tau = 1.

myGIF.h_Vhalf = -76.0
myGIF.h_k = -0.11
myGIF.h_tau = 50.

myGIF.n_Vhalf = -24.2
myGIF.n_k = 0.20
myGIF.n_tau = 100.

myGIF.E_K = -101.

#%% TEST SIMULATED SYNAPTIC INPUTS

myGIF.initializeSynapses(15, 0.010, 1, 20, 1000, 200, 50, 42)
#myGIF.plotSynaptic()

#%% TEST SIMULATION

time, V, m, h, n, spks = myGIF.simulateSynaptic(True)

plt.figure(figsize = (14, 10))
plt.plot(time, V)
plt.ylim(-77, 5)
plt.show()