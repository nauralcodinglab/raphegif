#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#%% UNPICKLE DATA

"""
latency_data.pyc was generated on Richard's cluster
"""

with open('./toy_IA_jitter/latency_data.pyc', 'rb') as f:
    d = pickle.load(f)

stds = d['stds']
tiled_ga_vec = d['tiled_ga_vec']
tiled_tau_h_vec = d['tiled_tau_h_vec']


#%% CREATE FIGURE

save_path = './figs/ims/'

plt.rc('text', usetex = True)

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.figure(figsize = (6, 6))

ax = plt.subplot(111, projection = '3d')
ax.set_title('Effect of $I_A$ on spike-time jitter in response to 25mV step')
ax.plot_surface(tiled_ga_vec, tiled_tau_h_vec, stds, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_xlabel('$\\bar{{g}}_a^\prime$')
ax.set_ylabel('$\\tau_h^\prime$')
ax.set_zlabel('Spike jitter ($\\tau_m$)')
ax.set_xlim3d(ax.get_xlim3d()[1], ax.get_xlim3d()[0])
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 15

if save_path is not None:
    plt.savefig(save_path + 'IA_jitter_3D.png', dpi = 300)

plt.show()
