#%% IMPORT MODULES

from __future__ import division

import multiprocessing as mp
import pickle

import numpy as np

from IAmod import IAmod, Simulation


#%% PERFORM SIMULATIONS

no_neurons = 1000
baseline_voltage = -60
pulse_ampli = 25
tau_h_prime = 1.5

no_points = 30

ga_vec = np.linspace(0, 15, num = no_points)
tau_h_vec = np.linspace(0.5, 2.5, num = no_points)


def generate_in_dict_ls(ga_vec, tau_h_vec):

    in_dict_ls = []

    for ga in ga_vec:
        for tau_h in tau_h_vec:
            in_dict_ls.append({'ga': ga, 'tau_h': tau_h})

    return in_dict_ls


def worker(in_dict):

    print('Simulating ga = {:.1f}, tau_h = {:.1f}'.format(in_dict['ga'], in_dict['tau_h']))

    tmp_mod = IAmod(in_dict['ga'], in_dict['tau_h'], 2)
    tmp_Vin = np.empty((3500, no_neurons), dtype = np.float64)
    tmp_Vin[:1000, :] = tmp_mod.ss_clamp(baseline_voltage)
    tmp_Vin[1000:, :] = tmp_mod.ss_clamp(baseline_voltage + pulse_ampli)

    tmp_sim = Simulation(tmp_mod, baseline_voltage, tmp_Vin)

    return np.nanstd(tmp_sim.get_spk_latencies())


in_dict_ls = generate_in_dict_ls(ga_vec, tau_h_vec)

if __name__ == '__main__':
    p = mp.Pool(20)

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


#%% EXPORT DATA

export_dict = {
    'stds': stds,
    'tiled_ga_vec': tiled_ga_vec,
    'tiled_tau_h_vec': tiled_tau_h_vec
}

with open('latency_data.pyc', 'wb') as f:
    pickle.dump(export_dict, f)
