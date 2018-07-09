#%% IMPORT MODULES

from __future__ import division

import multiprocessing as mp

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sys
sys.path.append('./toy_IA_jitter')
from IAmod import IAmod, Simulation


#%% PERFORM SIMULATIONS

no_neurons = 200
baseline_voltage = -60
pulse_ampli = 25
tau_h_prime = 1.5

no_points = 25

ga_vec = np.linspace(0, 15, num = no_points)
tau_h_vec = np.linspace(0.5, 2, num = no_points)


def generate_in_dict_ls(ga_vec, tau_h_vec):

    in_dict_ls = []

    for ga in ga_vec:
        for tau_h in tau_h_vec:
            in_dict_ls.append({'ga': ga, 'tau_h': tau_h})

    return in_dict_ls


def worker(in_dict):

    print('Simulating ga = {:.1f}, tau_h = {:.1f}'.format(in_dict['ga'], in_dict['tau_h']))

    tmp_mod = IAmod(in_dict['ga'], in_dict['tau_h'], 2)
    tmp_Vin = np.empty((6000, no_neurons), dtype = np.float64)
    tmp_Vin[:1000, :] = tmp_mod.ss_clamp(baseline_voltage)
    tmp_Vin[1000:, :] = tmp_mod.ss_clamp(baseline_voltage + pulse_ampli)

    tmp_sim = Simulation(tmp_mod, baseline_voltage, tmp_Vin)

    return np.nanstd(tmp_sim.get_spk_latencies())


in_dict_ls = generate_in_dict_ls(ga_vec, tau_h_vec)

if __name__ == '__main__':
    p = mp.Pool(8)

    latencies = p.map(worker, in_dict_ls)

    p.close()

print('Done!')

#%% PREPATE DATA FOR 3D PLOT

stds = np.empty((len(ga_vec), len(tau_h_vec)), dtype = np.float64)
tiled_ga_vec = np.tile(ga_vec[:, np.newaxis], (1, len(tau_h_vec)))
tiled_tau_h_vec = np.tile(tau_h_vec[np.newaxis, :], (len(ga_vec), 1))

cnt = 0
for i in range(len(ga_vec)):
    for j in range(len(tau_h_vec)):
        stds[i, j] = latencies[cnt]
        cnt += 1


#%% CREATE 3D PLOT

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
