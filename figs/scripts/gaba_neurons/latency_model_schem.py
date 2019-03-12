#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing/')

import OhmicSpkPredictor as osp
import src.pltools as pltools


#%% DEFINE FUNCTIONS FOR IF CELL

def voltage(t, tau, V0, Vinf):
    """Voltage of a leaky integrator in response to a step input starting at t = 0.
    """
    V_ = (V0 - Vinf) * np.exp(-t/tau) + Vinf
    V_ *= np.heaviside(t, 1)
    V_ += V0 * np.heaviside(-t, 0)
    return V_

def latency(tau, V0, Vinf, theta):
    """Spike latency for a leaky integrator in response to a step input starting at t = 0.
    (In other words, latency for V to cross theta.)
    """
    return -tau * np.log((theta - Vinf)/(V0 - Vinf))

def lif(t, tau, V0, Vinf, theta, spktop, Vreset, seed = 42, return_latency = False):

    dts = np.diff(t)
    dt = dts.mean()
    assert np.allclose(dts, dt)

    np.random.seed(seed)
    dV = lambda V_t, Vinf_t: (Vinf_t - V_t)/tau * dt + np.random.normal(size = 1) * np.sqrt(dt)

    Vout = np.empty_like(t)
    Vout[0] = V0

    latency_flag = False
    for i, t_ in enumerate(t):
        if i == 0:
            continue

        if t_ < 0:
            Vinf_t = V0
        else:
            Vinf_t = Vinf

        if Vout[i-1] < theta:
            Vout[i] = Vout[i - 1] + dV(Vout[i-1], Vinf_t)
        else:
            Vout[i-1] = spktop
            Vout[i] = Vreset
            if not latency_flag:
                latency = t_
                latency_flag = True

    if not return_latency:
        return Vout
    else:
        return Vout, latency


# Plotting helper functions
def I_vec(t, V0, Vinf):
    return np.ones_like(t_vec) * Vinf - (Vinf - V0) * np.heaviside(-t_vec, 0)

def scale(x, lower_bound = 0.2, upper_bound = 1):
    return x * (upper_bound - lower_bound) + lower_bound


#%% CREATE FIGURE
"""
Create a schematic to explain pre-pulse/spk latency experiments.
"""

IMG_PATH = './figs/ims/gaba_cells/'

Vinf = -40
V0s = [-70, -60, -50]
tau = 1
theta = -45
spktop = 0
Vreset = -60
t_vec = np.arange(-1, 3, 0.01)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

spec_outer = gs.GridSpec(1, 3)
spec_tr0 = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[:, 0], height_ratios = [1, 0.2])
spec_tr1 = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[:, 1], height_ratios = [1, 0.2])

plt.figure(figsize = (6, 3))

V_ax = plt.subplot(spec_tr0[0, :])
V_ax.set_title(r'\textbf{{A}}', loc = 'left')
V_ax.set_yticks([theta, Vinf])
V_ax.set_yticklabels([r'$\theta$', r'$V_\infty$'])
V_ax.set_xticks([])
V_ax.set_ylabel('$V$')
I_ax = plt.subplot(spec_tr0[1, :])
I_ax.set_yticks([])
I_ax.set_xticks([0])
I_ax.set_xticklabels(['$t_0$'])
I_ax.set_xlabel('Time')
#I_ax.set_ylabel('Input')
V_ax.plot(t_vec, lif(t_vec, tau, V0s[0], Vinf, theta, spktop, Vreset), 'k-', lw = 0.5, label = 'Simulated\ndata')
I_ax.plot(t_vec, I_vec(t_vec, V0s[0], Vinf), '-', color = 'gray', lw = 0.5)
V_ax.text(-0.5, V0s[0] + 2, '$V_0$', ha = 'center')
V_ax.axhline(theta, color = 'r', ls = '--', lw = 0.5, dashes = (5, 5))

pltools.hide_border('trb', ax = V_ax)
pltools.hide_border('trl', ax = I_ax)

lat_tmp = latency(tau, V0s[0], Vinf, theta)
V_ax.annotate('', (0, -30), (lat_tmp, -30), arrowprops = {'arrowstyle':'<->'})
V_ax.text((lat_tmp - 0)/2, -28, 'Latency', ha = 'center', va = 'bottom')
V_ax.legend(loc = 'upper left')

V_ax = plt.subplot(spec_tr1[0, :])
V_ax.set_title(r'\textbf{{B}}', loc = 'left')
V_ax.set_yticks([theta, Vinf])
V_ax.set_yticklabels([r'$\theta$', r'$V_\infty$'])
V_ax.set_xticks([])
#V_ax.set_ylabel('$V$')
I_ax = plt.subplot(spec_tr1[1, :])
I_ax.set_yticks([])
I_ax.set_xticks([0])
I_ax.set_xticklabels(['$t_0$'])
I_ax.set_xlabel('Time')
#I_ax.set_ylabel('Input')
for i, V0 in enumerate(V0s):
    sim_line = V_ax.plot(t_vec, voltage(t_vec, tau, V0, Vinf), 'r-', alpha = scale((len(V0s) - i)/len(V0s)))
    tr_line = V_ax.plot(t_vec, lif(t_vec, tau, V0, Vinf, theta, spktop, Vreset, 43 + i), 'k-', lw = 0.5)
    if i == 0:
        sim_line[0].set_label('Ohmic model')
        tr_line[0].set_label('Simulated data')
    I_ax.plot(t_vec, I_vec(t_vec, V0, Vinf), '-', color = 'gray', lw = 0.5, alpha = scale((len(V0s) - i)/len(V0s)))

V_ax.axhline(theta, color = 'r', ls = '--', lw = 0.5, dashes = (5, 5))

pltools.hide_border('trb', ax = V_ax)
pltools.hide_border('trl', ax = I_ax)

V_ax.legend()


V0_vec = np.arange(-75, -40, 0.1)
plt.subplot(spec_outer[:, 2])
plt.title(r'\textbf{{C}}', loc = 'left')
for i, V0 in enumerate(V0s):
    for z in range(5):
        Vrand = V0 + np.random.normal(0, 0.5, size = 1)
        pt = plt.plot(
            Vrand, lif(
                t_vec, tau, V0,
                Vinf, theta, spktop, Vreset, seed = 43 + i + z,
                return_latency = True
            )[1],
            'k.', alpha = 0.5
        )
        if i == 0 and z == 0:
            pt[0].set_label('Simulated\ntraces')
plt.plot(V0_vec, latency(tau, V0_vec, Vinf, theta), 'r-', label = 'Ohmic model')
plt.xlim(-78, -38)
plt.ylim(0, plt.ylim()[1])
plt.xticks([theta, Vinf], [r'$\theta$', r'$V_\infty$'])
plt.yticks([0, 1, 2], ['0', '1', '2'])
plt.xlabel('$V_0$')
plt.ylabel('Latency (a.u.)')
pltools.hide_border('tr')
plt.legend()

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'latency_expt_sketch.png')

plt.show()
