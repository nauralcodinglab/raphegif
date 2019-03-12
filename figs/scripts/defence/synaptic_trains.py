#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./figs/scripts/defence')

sys.path.append('./analysis/spk_timing/IA_mod')

from stimgen import Stim
import IAmod
import src.pltools as pltools


#%% INITIALIZE MODELS

noise_sd = 0

linmod = IAmod.IAmod(0, 1, noise_sd)
linmod.El = -70
iamod = IAmod.IAmod(10, 1, noise_sd)
iamod.El = -70


#%% CREATE SYNAPTIC BOMBARDMENT CLASS WITH HANDY METHODS

class SynapticBombardment(object):

    def __init__(self, dt = 0.001):
        self.dt = dt

    def set_kernel(self, kernel, kernel_params = None):
        self.kernel = kernel
        self.kernel_params = kernel_params

    def generate_kernel(self, half_T, ampli, tau_rise, tau_decay):
        """Generate a synaptic kernel.

        Inputs:
            half_T      --  Half width of the kernel (so rise is centred)
            ampli       --  Peak amplitude
            tau_rise
            tau_decay
        """

        t_vec = np.arange(-half_T, half_T, self.dt)

        kernel = np.heaviside(t_vec, 1) * (np.exp(-t_vec/tau_decay) - np.exp(-t_vec/tau_rise))
        kernel /= np.max(kernel)
        kernel *= ampli

        # Assign output to self.
        self.kernel_params = {
            'ampli': ampli,
            'tau_rise': tau_rise,
            'tau_decay': tau_decay
        }
        self.kernel = kernel

        return kernel

    def poisson_rtp(self, rate):
        """Convert the rate of a Poisson point process to a probability."""
        return 1 - np.exp(-rate * self.dt)

    def set_trigger(self, trigger):

        if trigger.ndim < 2:
            trigger = (1 * trigger)[:, np.newaxis]

        self.trigger_mat = trigger

    def generate_trigger(self, rate, T, no_neurons, seed = 42):
        """Generate a sparse matrix with ones placed according to a Poisson point process."""

        trigger_mat = np.zeros((int(T/self.dt), no_neurons))
        np.random.seed(seed)
        rands = np.random.uniform(0, 1, trigger_mat.shape)
        trigger_mat[self.poisson_rtp(rate) >= rands] = 1

        self.trigger_mat = trigger_mat

        return trigger_mat

    def convolve(self):
        """Convolve trigger matrix with synaptic kernel."""

        I = np.empty_like(self.trigger_mat)

        for i in range(self.trigger_mat.shape[1]):
            I[:, i] = np.convolve(self.trigger_mat[:, i], self.kernel, 'same')

        self.I = I

        return I

    @property
    def no_neurons(self):
        try:
            no_neurons = self.trigger.shape[1]
        except AttributeError:
            no_neurons = self.I.shape[1]

        return no_neurons

    @property
    def t_vec(self):
        return np.arange(0, (self.I.shape[0] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self):
        return np.tile(self.t_vec[:, np.newaxis], (1, self.no_neurons))

    def plot(self):
        plt.figure()
        plt.plot(self.t_mat, self.I, 'k-', alpha = min(1, 3 / self.no_neurons))
        plt.xlabel('Time')
        plt.show()


#%% GENERATE SYNAPTIC KERNELS

# Normalize timescales to membrane tau for use by IAmod.
membrane_tau = 75
dt = 0.01

lhb_ampli = 80
mpfc_ampli = 20
rise = 1 / membrane_tau
lhb_decay = 10 / membrane_tau
mpfc_decay = 150 / membrane_tau

tmp_syn = SynapticBombardment(dt)
mpfc_kernel = tmp_syn.generate_kernel(4.5 * mpfc_decay, mpfc_ampli, rise, mpfc_decay)
lhb_kernel = tmp_syn.generate_kernel(6 * lhb_decay, lhb_ampli, rise, lhb_decay)

10/75
#%% SIMPLE SIMULATIONS TO INSPECT KERNELS

IMG_PATH = './figs/ims/defence/'

T_single_epsc = 20

trigger_vec = np.zeros(int(T_single_epsc/dt))
trigger_vec[100] = 1

mpfc_single_epsc = SynapticBombardment(dt)
mpfc_single_epsc.set_kernel(mpfc_kernel)
mpfc_single_epsc.set_trigger(trigger_vec)
mpfc_single_epsc.convolve()

lhb_single_epsc = SynapticBombardment(dt)
lhb_single_epsc.set_kernel(lhb_kernel)
lhb_single_epsc.set_trigger(trigger_vec)
lhb_single_epsc.convolve()

mpfc_single_lin     = IAmod.Simulation(linmod, -70, mpfc_single_epsc.I * 2.25, dt)
lhb_single_lin      = IAmod.Simulation(linmod, -70, lhb_single_epsc.I * 2.25, dt)
mpfc_single_ia      = IAmod.Simulation(iamod, -70, mpfc_single_epsc.I * 2.25, dt)
lhb_single_ia       = IAmod.Simulation(iamod, -70, lhb_single_epsc.I * 2.25, dt)

"""lhb_single_ia.simple_plot()
mpfc_single_ia.simple_plot()"""

AUC_mpfc = (mpfc_single_lin.V - linmod.El).sum() * dt
AUC_lhb = (lhb_single_lin.V - linmod.El).sum() * dt


spec_outer = gs.GridSpec(2, 2, height_ratios = [0.3, 1], hspace = 0)

plt.style.use('./figs/scripts/defence/defence_mplrc.dms')
plt.figure(figsize = (5, 3))

plt.subplot(spec_outer[0, 0])
plt.title('mPFC-like input (slow EPSC decay)')
plt.plot(mpfc_single_epsc.t_vec, mpfc_single_epsc.I, '-', color = 'gray', lw = 0.7)
plt.xticks([])
plt.xlim(0, 8)
plt.ylabel('Input (mV)')

plt.subplot(spec_outer[1, 0])
plt.plot(mpfc_single_lin.t_vec, mpfc_single_lin.V[:, 0], 'r-', label = 'Linear model')
plt.plot(mpfc_single_ia.t_vec, mpfc_single_ia.V[:, 0], 'b-', alpha = 0.9, label = 'Linear + $I_A$')
plt.xlim(0, 8)
plt.ylabel('V (mV)')
plt.xlabel('Time ($\\tau_\\mathrm{{mem}}$)')
plt.legend()

plt.subplot(spec_outer[0, 1])
plt.title('LHb-like input (fast EPSC decay)')
plt.plot(lhb_single_epsc.t_vec, lhb_single_epsc.I, '-', color = 'gray', lw = 0.7)
plt.xticks([])
plt.xlim(0, 8)
plt.ylabel('Input (mV)')

plt.subplot(spec_outer[1, 1])
plt.plot(lhb_single_lin.t_vec, lhb_single_lin.V[:, 0], 'r-', label = 'Linear model')
plt.plot(lhb_single_ia.t_vec, lhb_single_ia.V[:, 0], 'b-', alpha = 0.9, label = 'Linear + $I_A$')
plt.xlim(0, 8)
plt.ylabel('V (mV)')
plt.xlabel('Time ($\\tau_\\mathrm{{mem}}$)')
plt.legend()

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'synpatic_kernels.png')

plt.show()

print('mPFC EPSP AUC: {:.2f}mV tau'.format(AUC_mpfc))
print('LHb EPSP AUC: {:.2f}mV tau'.format(AUC_lhb))


#%% SIMULATE RESPONSE TO SYNAPTIC BOMBARDMENT

T_bombardment = 20 / (75 * dt)
no_neurons = 200

lhb_rate = 0.015 * membrane_tau
mpfc_rate = lhb_rate * AUC_lhb / AUC_mpfc

lhb_poisson = SynapticBombardment(dt)
lhb_poisson.set_kernel(lhb_kernel)
lhb_poisson.generate_trigger(lhb_rate, T_bombardment, no_neurons)
lhb_poisson.convolve()
#lhb_poisson.plot()

mpfc_poisson = SynapticBombardment(dt)
mpfc_poisson.set_kernel(mpfc_kernel)
mpfc_poisson.generate_trigger(mpfc_rate, T_bombardment, no_neurons)
mpfc_poisson.convolve()
#mpfc_poisson.plot()

print('Performing linear simulations...')
lhb_bbd_lin_sim = IAmod.Simulation(linmod, -70, lhb_poisson.I, dt)
mpfc_bbd_lin_sim = IAmod.Simulation(linmod, -70, mpfc_poisson.I, dt)

print('Performing IA simulations...')
lhb_bbd_ia_sim = IAmod.Simulation(iamod, -70, lhb_poisson.I, dt)
mpfc_bbd_ia_sim = IAmod.Simulation(iamod, -70, mpfc_poisson.I, dt)
print('Done!')

#lhb_bbd_ia_sim.simple_plot()
#mpfc_bbd_ia_sim.simple_plot()

#%% GENERATE FIGURE

IMG_PATH = './figs/ims/defence/'

def plot_raster(sim, color, neurons_to_show, ax = None):
    if ax is None:
        ax = plt.gca()

    for i in range(neurons_to_show):
        spk_times_tmp = np.where(sim.spks[:, i])[0] * sim.dt
        plt.plot(
            spk_times_tmp, [i for j in range(len(spk_times_tmp))],
            '|', color = color, markersize = 2
        )

linred = (0.9, 0.2, 0.2)
iablue = (0.2, 0.2, 0.8)

neurons_to_show = 20

spec_outer = gs.GridSpec(1, 2)
spec_lhb            = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[:, 0], height_ratios = [0.4, 1, 0.8], hspace = 0)
spec_lhb_raster     = gs.GridSpecFromSubplotSpec(2, 1, spec_lhb[2, :], hspace = 0)
spec_mpfc           = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[:, 1], height_ratios = [0.4, 1, 0.8], hspace = 0)
spec_mpfc_raster    = gs.GridSpecFromSubplotSpec(2, 1, spec_mpfc[2, :], hspace = 0)

plt.figure(figsize = (5, 3))

### LHb side
plt.subplot(spec_lhb[0, :])
plt.title('LHb-like input (fast EPSC decay)')
plt.plot(
    lhb_poisson.t_vec, lhb_poisson.I[:, 0],
    '-', color = 'gray', lw = 0.7
)
plt.xlim(5, T_bombardment)
plt.xticks([])
plt.yticks([])
pltools.hide_border()

plt.subplot(spec_lhb[1, :])
plt.plot(
    lhb_bbd_lin_sim.t_vec, lhb_bbd_lin_sim.V[:, 0],
    '-', color = linred, lw = 0.7, label = 'Linear model'
)
plt.plot(
    lhb_bbd_ia_sim.t_vec, lhb_bbd_ia_sim.V[:, 0],
    '-', color = iablue, lw = 0.7, alpha = 0.8, label = 'Linear + $I_A$'
)
plt.xlim(5, T_bombardment)
plt.xticks([])
pltools.hide_border('trb')
plt.ylabel('V (mV)')
plt.legend()

plt.subplot(spec_lhb_raster[0, :])
plot_raster(lhb_bbd_lin_sim, linred, neurons_to_show)
plt.xlim(5, T_bombardment)
plt.yticks([])
plt.xticks([])
pltools.hide_border()

plt.subplot(spec_lhb_raster[1, :])
plot_raster(lhb_bbd_ia_sim, iablue, neurons_to_show)
plt.xlim(5, T_bombardment)
plt.yticks([])
pltools.hide_border('trl')
plt.xlabel('Time ($\\tau_\\mathrm{{mem}}$)')


### mPFC side
plt.subplot(spec_mpfc[0, :])
plt.title('mPFC-like input (slow EPSC decay)')
plt.plot(
    mpfc_poisson.t_vec, mpfc_poisson.I[:, 0],
    '-', color = 'gray', lw = 0.7
)
plt.xlim(5, T_bombardment)
plt.xticks([])
plt.yticks([])
pltools.hide_border()

plt.subplot(spec_mpfc[1, :])
plt.plot(
    mpfc_bbd_lin_sim.t_vec, mpfc_bbd_lin_sim.V[:, 0],
    '-', color = linred, lw = 0.7
)
plt.plot(
    mpfc_bbd_ia_sim.t_vec, mpfc_bbd_ia_sim.V[:, 0],
    '-', color = iablue, lw = 0.7, alpha = 0.8
)
plt.xlim(5, T_bombardment)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec_mpfc_raster[0, :])
plot_raster(mpfc_bbd_lin_sim, linred, neurons_to_show)
plt.xlim(5, T_bombardment)
plt.yticks([])
plt.xticks([])
pltools.hide_border()

plt.subplot(spec_mpfc_raster[1, :])
plot_raster(mpfc_bbd_ia_sim, iablue, neurons_to_show)
plt.xlim(5, T_bombardment)
plt.yticks([])
pltools.hide_border('trl')
plt.xlabel('Time ($\\tau_\\mathrm{{mem}}$)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'synaptic_bombardment.png')

plt.show()


#%%

def extract_ISIs(sim):

    ISIs = []

    for i in range(sim.spks.shape[1]):
        spk_times_tmp = np.where(sim.spks[:, i])[0] * sim.dt
        if len(spk_times_tmp) > 1:
            ISIs.extend(spk_times_tmp[1:] - spk_times_tmp[:-1])

    return np.array(ISIs)

tmp = extract_ISIs(mpfc_bbd_ia_sim)

plt.figure(figsize = (5, 3))

plt.subplot(121)
plt.title('LHb-like input ISI distribution')
plt.hist(extract_ISIs(lhb_bbd_lin_sim), color = linred, label = 'Linear model')
plt.hist(extract_ISIs(lhb_bbd_ia_sim), color = iablue, alpha = 0.7, label = 'Linear + $I_A$')
plt.xlabel('ISI ($\\tau_\\mathrm{mem}$)')
pltools.hide_border('tr')

plt.subplot(122)
plt.title('mPFC-like input ISI distribution')
plt.hist(extract_ISIs(mpfc_bbd_lin_sim), color = linred, label = 'Linear model')
plt.hist(extract_ISIs(mpfc_bbd_ia_sim), color = iablue, alpha = 0.7, label = 'Linear + $I_A$')
plt.xlabel('ISI ($\\tau_\\mathrm{mem}$)')
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'LHb_vs_mPFC_ISI_dist.png')

plt.show()


### INFORMATION THEORETIC STUFF BEYOND HERE
"""
Below here I started to work on computing the SR coherence under different conditions.
"""

#%%

T_bombardment = 10 * 1e3/75

t_vec = np.arange(0, T_bombardment, dt)[:int(T_bombardment/dt)]

sin_slow = 0.005 * np.sin(2 * np.pi * t_vec / 4.5)
sin_fast = 0.005 * np.sin(2 * np.pi * t_vec / 1)

tmp = generate_poisson_trigger(lhb_rate + sin_slow + sin_fast, T_bombardment, dt)
tmp2 = np.convolve(tmp, lhb_kernel, 'same')[:, np.newaxis]

plt.figure()

plt.subplot(211)
plt.plot(sin_slow + sin_fast)

plt.subplot(212)
plt.plot(tmp2)
plt.show()

#%%

tmp_spks = 1* mpfc_bbd_lin_sim.spks.astype(np.float64).flatten()
tmp_stim = 1* lhb_vin.flatten()

tmp_stim -= tmp_stim.mean()
tmp_spks -= 0.5#tmp_spks.mean()

tmp_spks_f = np.fft.rfft(tmp_spks)
tmp_stim_f = np.fft.rfft(tmp_stim)

plt.plot(np.abs(tmp_spks_f * tmp_stim_f.conj()) **2)

%matplotlib qt5
plt.plot(np.abs(tmp_spks_f)**2)

plt.plot(np.abs(tmp_stim_f)**2)

plt.plot(np.abs(tmp_spks_f)**2 * np.abs(tmp_stim_f)**2)

#%%

def sr_coherence(stimulus, response, dt):

    stimulus = deepcopy(stimulus)
    response = deepcopy(response)

    stimulus -= stimulus.mean()
    response -= response.mean()

    stim_f = np.fft.rfft(stimulus)
    resp_f = np.fft.rfft(response)

    coherence = np.abs(resp_f * stim_f.conj()) **2 / (np.abs(resp_f)**2 * np.abs(stim_f)**2)
    f_bins = np.fft.rfftfreq(len(stimulus), dt)

    return coherence, f_bins

coh, fbins = sr_coherence(lhb_vin.flatten(), tmp_spks, dt)


plt.figure()
plt.plot(fbins, coh)
plt.show()
