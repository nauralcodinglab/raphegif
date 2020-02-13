#%% IMPORT MODULES

from __future__ import division

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from grr.Simulation import GIFnet_Simulation
from grr.Spiketrain import PSTH

#%% LOAD DATA

DATA_PATH = os.path.join('data', 'simulations', 'GIF_network')
simfiles = {}
for fname in os.listdir(DATA_PATH):
    if fname[-4:].lower() == 'hdf5':
        try:
            simfiles[fname[:-5]] = GIFnet_Simulation(
                os.path.join(DATA_PATH, fname), mode = 'r'
            )
        except IOError:
            warnings.warn(
                'Problem with file {}. Continuing.'.format(fname),
                RuntimeWarning
            )
    else:
        continue

#%% HANDY FUNCTIONS

def plot_ser_traces(spec, sim_obj, step_no, title = ''):

    spec_tr = gs.GridSpecFromSubplotSpec(
        3, 1, spec,
        height_ratios = [0.3, 1., 0.6],
        hspace = 0.05
    )

    plt.subplot(spec_tr[0,:])
    plt.title(title)
    plt.plot(
        sim_obj.get_ser_examples_supp()[step_no, ...].T,
        sim_obj.ser_examples['I'][step_no, ...].T,
        '-', color = 'gray', alpha = 0.8, lw = 0.5
    )
    plt.xticks([])

    plt.subplot(spec_tr[1,:])
    plt.plot(
        sim_obj.get_ser_examples_supp()[step_no, ...].T,
        sim_obj.ser_examples['V'][step_no, ...].T,
        '-', color = 'k', alpha = 0.8, lw = 0.5
    )
    plt.xticks([])

    ax_psth = plt.subplot(spec_tr[2, :])
    ax_psth.plot(
        sim_obj.get_t_vec(),
        PSTH(
            sim_obj.ser_spktrains[step_no, ...].sum(axis = 0),
            50., 600, 0.1
        ),
        'g-'
    )
    plt.xlabel('Time (ms)')
    ax_event = ax_psth.twinx()
    ax_event.eventplot(
        sim_obj.get_ser_spktimes()[step_no],
        color = 'k', linelengths = 3
    )
    ax_event.set_yticks([])

def plot_gaba_traces(spec, sim_obj, step_no, title = ''):

    spec_tr = gs.GridSpecFromSubplotSpec(
        3, 1, spec,
        height_ratios = [0.3, 1., 0.6],
        hspace = 0.05
    )

    plt.subplot(spec_tr[0,:])
    plt.title(title)
    plt.plot(
        sim_obj.get_gaba_examples_supp()[step_no, ...].T,
        sim_obj.gaba_examples['I'][step_no, ...].T,
        '-', color = 'gray', alpha = 0.8, lw = 0.5
    )
    plt.xticks([])

    plt.subplot(spec_tr[1,:])
    plt.plot(
        sim_obj.get_gaba_examples_supp()[step_no, ...].T,
        sim_obj.gaba_examples['V'][step_no, ...].T,
        '-', color = 'k', alpha = 0.8, lw = 0.5
    )
    plt.xticks([])

    ax_psth = plt.subplot(spec_tr[2, :])
    ax_psth.plot(
        sim_obj.get_t_vec(),
        PSTH(
            sim_obj.gaba_spktrains[step_no, ...].sum(axis = 0),
            50., 600, 0.1
        ),
        'g-'
    )
    plt.xlabel('Time (ms)')
    ax_event = ax_psth.twinx()
    ax_event.eventplot(
        sim_obj.get_gaba_spktimes()[step_no],
        color = 'k', linelengths = 3
    )
    ax_event.set_yticks([])

#%% PLOT CHANNELS

try:

    IMG_PATH = os.path.join('figs', 'ims', 'GIF_network')
    sweep_no = 8

    plt.figure(figsize = (8,6))

    spec_outer = gs.GridSpec(
        2, 1,
        hspace = 0.4, top = 0.9, right = 0.95, left = 0.1, bottom = 0.1
    )
    spec_gr = gs.GridSpecFromSubplotSpec(
        1, 3, spec_outer[0, :], wspace = 0.4
    )
    spec_ngr = gs.GridSpecFromSubplotSpec(
        1, 3, spec_outer[1, :], wspace = 0.4
    )

    plot_ser_traces(spec_gr[:, 0], simfiles['subsample_base_l_g'], sweep_no, 'Base + GABA')
    plot_ser_traces(spec_gr[:, 1], simfiles['subsample_fixedIA_l_g'], sweep_no, 'Fixed $I_A$ $+$ GABA')
    plot_ser_traces(spec_gr[:, 2], simfiles['subsample_noIA_l_g'], sweep_no, '$I_A$ KO $+$ GABA')

    plot_ser_traces(spec_ngr[:, 0], simfiles['subsample_base_l_ng'], sweep_no, 'Base $-$ GABA')
    plot_ser_traces(spec_ngr[:, 1], simfiles['subsample_fixedIA_l_ng'], sweep_no, 'Fixed $I_A$ $-$ GABA')
    plot_ser_traces(spec_ngr[:, 2], simfiles['subsample_noIA_l_ng'], sweep_no, '$I_A$ KO $-$ GABA')

    plt.tight_layout()

    if IMG_PATH is not None:
        plt.savefig(os.path.join(IMG_PATH, 'traces.png'), dpi = 300)

except:
    for key in simfiles.keys():
        simfiles[key].close()
    raise

#%% PLOT CHANNELS FOR GABA CELLS

try:
    sweep_no = 8

    plt.figure(figsize = (8,6))

    spec_outer = gs.GridSpec(
        1, 1,
        hspace = 0.4, top = 0.9, right = 0.95, left = 0.1, bottom = 0.1
    )
    spec_gr = gs.GridSpecFromSubplotSpec(
        1, 3, spec_outer[0, :], wspace = 0.4
    )

    plot_gaba_traces(spec_gr[:, 0], simfiles['subsample_base_l_g'], sweep_no, 'Base + GABA')
    plot_gaba_traces(spec_gr[:, 1], simfiles['subsample_fixedIA_l_g'], sweep_no, 'Fixed $I_A$ $+$ GABA')
    plot_gaba_traces(spec_gr[:, 2], simfiles['subsample_noIA_l_g'], sweep_no, '$I_A$ KO $+$ GABA')

    plt.tight_layout()

    if IMG_PATH is not None:
        plt.savefig(os.path.join(IMG_PATH, 'gaba_traces.png'), dpi = 300)

except:
    for key in simfiles.keys():
        simfiles[key].close()
    raise

#%% CLOSE ALL FILES

for key in simfiles.keys():
    simfiles[key].close()
