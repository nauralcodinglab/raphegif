#%% IMPORT MODULES

from __future__ import division

import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%% CREATE FIGURE

"""
latency_ampli_data.pyc dataset was generated on Richard's Linux cluster.
"""

with open('./toy_IA_jitter/latency_ampli_data.pyc', 'rb') as f:
    d = pickle.load(f)

stds = d['stds']
tiled_ga_vec = d['tiled_ga_vec']
tiled_ampli_vec = d['tiled_ampli_vec']

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

plt.figure(figsize = (10, 5))

plt.suptitle('Effect of input strength on spike jitter for $\\tau_h^\prime = 1.5$, $V_\mathrm{{rest}} = -60\mathrm{{mV}}$, $\\theta = -45\mathrm{{mV}}$')

raw_ax = plt.subplot(121, projection = '3d')
raw_ax.set_title('Raw jitter')
raw_ax.plot_surface(
    tiled_ga_vec, tiled_ampli_vec, stds,
    rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0,
    antialiased = False
)
raw_ax.set_xlabel('$\\bar{{g}}_a^\prime$')
raw_ax.set_ylabel('Step size (mV)')
raw_ax.set_zlabel('Spike jitter ($\\tau_m$)')
raw_ax.set_xlim3d(raw_ax.get_xlim3d()[1], raw_ax.get_xlim3d()[0])
raw_ax.set_ylim3d(raw_ax.get_ylim3d()[1], raw_ax.get_ylim3d()[0])
raw_ax.xaxis.labelpad = 15
raw_ax.yaxis.labelpad = 15
raw_ax.zaxis.labelpad = 15

normed_ax = plt.subplot(122, projection = '3d')
normed_ax.set_title('Jitter relative to $\\bar{{g}}_a^\prime = 0$')
normed_ax.plot_surface(
    tiled_ga_vec, tiled_ampli_vec, stds / stds[0, :],
    rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0,
    antialiased = False
)
normed_ax.set_xlabel('$\\bar{{g}}_a^\prime$')
normed_ax.set_ylabel('Step size (mV)')
normed_ax.set_zlabel('Normalized jitter (fold change)')
normed_ax.set_xlim3d(normed_ax.get_xlim3d()[1], normed_ax.get_xlim3d()[0])
normed_ax.set_ylim3d(normed_ax.get_ylim3d()[1], normed_ax.get_ylim3d()[0])
normed_ax.xaxis.labelpad = 15
normed_ax.yaxis.labelpad = 15
normed_ax.zaxis.labelpad = 15

plt.subplots_adjust(left = 0.025, right = 0.9, top = 0.85, bottom = 0.15)

if save_path is not None:
    plt.savefig(save_path + 'IA_ampli_jitter_3D.png', dpi = 300)

plt.show()
