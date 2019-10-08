#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing/')

import OhmicSpkPredictor as osp
from grr.cell_class import Cell
from grr import pltools


#%% LOAD DATA

DATA_PATH = './data/raw/GABA/spk_time/'

inventory = pd.read_csv(DATA_PATH + 'index.csv')
inventory['cumcount'] = inventory.groupby('Cell_ID').cumcount()
fnames = inventory.pivot('Cell_ID', 'cumcount', values = 'Recording').values.tolist()
fnames = [[f for f in fs if f is not None] for fs in fnames] # Remove `None`s

cellnames = inventory.pivot('Cell_ID', 'cumcount', values = 'Recording').index.tolist()

# Manually reject recordings.
rejects = ['18711002.abf']

taus = []
cells = []
for i in range(len(fnames)):
    ce_tmp = Cell().read_ABF([DATA_PATH + fname for fname in fnames[i] if fname not in rejects])
    taus_tmp = []
    for rec in ce_tmp:
        rec.set_dt(0.1)
        out = rec.fit_test_pulse(
            (0, 100), (5000, 5100), V_clamp = False, tau = (1650, 5000),
            V_chan = 0, I_chan = 1,
            verbose = False, plot_tau = True)
        taus_tmp.append(out['tau'])
    print('Tau: {:.1f} +- {:.1f}ms'.format(np.mean(taus_tmp), np.std(taus_tmp)))
    taus.append(np.mean(taus_tmp))
    cells.append(ce_tmp)



#%%

cells[0][0].plot()
cells[0][0].set_dt(0.1)
cells[0][0].fit_test_pulse((0, 100), (5000, 5100), V_clamp = False, tau = (1620, 1800, 3000))

#%%
test_cell = cells[0]
test_cell[0].shape
pred = osp.OhmicSpkPredictor()
pred.add_recordings(test_cell, (0, 100), (5000, 5100))
pred.scrape_data(quiescent_until = 2662)
pred.spks
pred.V0

plt.figure()
plt.plot(pred.V0, pred.spks, 'ko')

#%% SCRAPE SPIKE LATENCIES

predictors = []

for tau, ce in zip(taus, cells):
    pred_tmp = osp.OhmicSpkPredictor()
    pred_tmp.add_recordings(ce, (0, 100), (5000, 5100))
    pred_tmp.scrape_data(quiescent_until = 2662)
    pred_tmp.fit_spks(force_tau = tau, Vinf_guesses = [pred_tmp.Vinf.mean()], thresh_guesses = np.linspace(-100, 200, 600))

    predictors.append(pred_tmp)



#%% INSPECT SCRAPED SPIKE LATENCIES
plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

tr_xlim = slice(16000, 33000)

#spec_outer = gs.GridSpec(len(cells), )



for id, ce, pr in zip(cellnames, cells, predictors):

    plt.figure(figsize = (6, 3))

    plt.subplot(121)
    plt.title('{}'.format(id))
    plt.plot(ce[0][0, tr_xlim, ::4], 'k-', lw = 0.5)
    plt.xticks([])

    plt.subplot(122)
    plt.plot(pr.V0, pr.spks, 'ko', label = 'Observed spikes')
    V0s = np.linspace(-110, -40, 200)
    plt.plot(
        V0s, pr.predict_spks(V0 = V0s, Vinf = pr.Vinf_est),
        'r--', label = 'Ohmic model\n$\\hat{{\\theta}} = $ {:.1f}mV'.format(pr.thresh)
    )
    plt.plot(
        V0s, pr.predict_spks(V0 = V0s, Vinf = pr.Vinf_est, thresh = -45),
        'g--', label = 'Ohmic model\n$\\theta$ fixed at $-45$mV'
    )
    plt.axvline(pr.Vinf_est, color = 'k', lw = 0.5)
    """plt.text(
        0.9, 0.9, r'$\hat{{\theta}}$ = {:.1f}mV'.format(pr.thresh),
        ha = 'right', va = 'top', transform = plt.gca().transAxes)"""
    plt.ylim(0, plt.ylim()[1])
    plt.xlim(-119, -37)
    plt.ylabel('Latency (ms)')
    plt.xlabel('$V_0$ (mV)')
    plt.legend()

    plt.tight_layout()
    plt.show()

#%%

IMG_PATH = './figs/ims/gaba_cells/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

tr_xrange = slice(15000, 30000)

tr_params = {
    'sweeps': [3, -2, -1],
    'cols': ['k', 'k', 'k'],
    'alphas': [1, 0.5, 0.3]
}

def plot_traces(rec, primary_ax, secondary_ax, param_dict, primary_channel = 0, secondary_channel = 1, dt = 0.1):

    sweeps  = param_dict['sweeps']
    cols    = param_dict['cols']
    alphas  = param_dict['alphas']

    if not all([len(sweeps) == len(x) for x in [cols, alphas]]):
        raise ValueError('sweeps, cols, and alphas not of identical lengths.')

    t_vec = np.arange(0, (rec.shape[1] - 0.5) * dt, dt)

    if primary_ax is not None:
        for sw, col, alph in zip(sweeps, cols, alphas):
            primary_ax.plot(
                t_vec, rec[primary_channel, :, sw],
                '-', color = col, lw = 0.5, alpha = alph
            )
    if secondary_ax is not None:
        for sw, col, alph in zip(sweeps, cols, alphas):
            secondary_ax.plot(
                t_vec, rec[secondary_channel, :, sw],
                '-', color = 'gray', lw = 0.5, alpha = alph
            )

spec_outer = gs.GridSpec(len(cells), 2)

plt.figure(figsize = (6, 8))

Letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(cells)):

    cename   = cellnames[i]
    ce      = cells[i]
    pr      = predictors[i]

    spec_I_tmp = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[i, 0], height_ratios = [1, 0.2], hspace = 0)

    I_Vax_tmp = plt.subplot(spec_I_tmp[0, :])
    I_Vax_tmp.set_title(r'\textbf{{{}1}} {} current steps'.format(Letters[i], cename), loc = 'left')
    I_Vax_tmp.set_xticks([])
    I_Vax_tmp.set_ylabel('$V$ (mV)')
    I_Vax_tmp.set_ylim(-105, 10)
    I_Vax_tmp.set_yticks([-100, -50, 0])
    pltools.hide_border('trb', I_Vax_tmp)
    I_Iax_tmp = plt.subplot(spec_I_tmp[1, :])
    I_Iax_tmp.set_yticks([])
    pltools.hide_border('ltr', I_Iax_tmp)
    if i == (len(cells) - 1):
        I_Iax_tmp.set_xlabel('Time (ms)')
    else:
        I_Iax_tmp.set_xticks([])
        pltools.hide_border('b', I_Iax_tmp)

    plot_traces(ce[0][:, tr_xrange, :], I_Vax_tmp, I_Iax_tmp, tr_params)
    I_Vax_tmp.axhline(pr.thresh, color = 'r', lw = 0.5, ls = '--', dashes = (5, 5))
    if i == 0:
        I_Vax_tmp.annotate(
            r'$\hat{{\theta}}$',
            (120, pr.thresh), xytext = (20, 0), ha = 'center', va = 'top',
            arrowprops = {'arrowstyle':'->'}
        )

    plt.subplot(spec_outer[i, 1])
    plt.title(r'\textbf{{{}2}} {} spike latencies'.format(Letters[i], cename), loc = 'left')
    plt.plot(pr.V0, pr.spks, 'ko')
    V0s = np.linspace(-110, -40, 200)
    plt.plot(
        V0s, pr.predict_spks(V0 = V0s, Vinf = pr.Vinf_est),
        'r-', label = 'Ohmic model\n$\\hat{{\\theta}} = {:.1f}$mV'.format(pr.thresh)
    )
    if i == 0:
        plt.annotate(
            r'$\hat{{\theta}}$',
            (pr.thresh, 0), (-70, 20),
            ha = 'center', va = 'bottom',
            arrowprops = {'arrowstyle': '->'}
        )
        plt.annotate(
            r'$V_\infty$',
            (pr.Vinf_est, 130), (-60, 175),
            ha = 'center', va = 'center',
            arrowprops = {'arrowstyle': '->'}
        )
    plt.axvline(pr.Vinf_est, color = 'k', lw = 0.5)
    plt.ylim(0, plt.ylim()[1])
    plt.xlim(-119, -37)
    plt.ylabel('Latency (ms)')
    if i == (len(cells) - 1):
        plt.xlabel('$V_0$ (mV)')
    plt.legend()
    pltools.hide_border('tr')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'steps_latencies.png')

plt.show()
