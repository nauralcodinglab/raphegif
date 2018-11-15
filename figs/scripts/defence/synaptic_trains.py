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


#%% INITIALIZE MODELS

noise_sd = 0

linmod = IAmod.IAmod(0, 1, noise_sd)
iamod = IAmod.IAmod(10, 1, noise_sd)


#%% GENERATE SYNAPTIC KERNELS

def generate_synaptic_kernel(half_T, ampli, tau_rise, tau_decay, dt = 0.001):

    t_vec = np.arange(-half_T, half_T, dt)

    waveform = np.heaviside(t_vec, 1) * (np.exp(-t_vec/tau_decay) - np.exp(-t_vec/tau_rise))
    waveform /= np.max(waveform)
    waveform *= ampli

    return waveform

# Normalize timescales to membrane tau for use by IAmod.
membrane_tau = 75
dt = 0.01

lhb_ampli = 40
mpfc_ampli = 12.5
rise = 1 / membrane_tau
lhb_decay = 10 / membrane_tau
mpfc_decay = 75 / membrane_tau

mpfc_kernel = generate_synaptic_kernel(6 * mpfc_decay, mpfc_ampli, rise, mpfc_decay, dt)
lhb_kernel = generate_synaptic_kernel(6 * lhb_decay, lhb_ampli, rise, lhb_decay, dt)


#%% SIMPLE SIMULATIONS TO INSPECT KERNELS

T_single_epsc = 10

trigger_vec = np.zeros(int(15/dt))
trigger_vec[100] = 1

mpfc_single_epsc = np.convolve(trigger_vec, mpfc_kernel, 'same')[:, np.newaxis]
lhb_single_epsc = np.convolve(trigger_vec, lhb_kernel, 'same')[:, np.newaxis]

mpfc_single_sim = IAmod.Simulation(linmod, -60, mpfc_single_epsc, dt)
lhb_single_sim = IAmod.Simulation(linmod, -60, lhb_single_epsc, dt)

AUC_mpfc = (mpfc_single_sim.V - linmod.El).sum() * dt
AUC_lhb = (lhb_single_sim.V - linmod.El).sum() * dt


plt.style.use('./figs/scripts/defence/defence_mplrc.dms')
plt.figure(figsize = (3, 2))

I_ax = plt.subplot(211)
plt.plot(mpfc_single_sim.t_vec, mpfc_single_epsc[:, 0], 'k-', label = 'mPFC')
plt.plot(lhb_single_sim.t_vec, lhb_single_epsc[:, 0], 'r-', alpha = 0.9, label = 'LHb')
plt.xticks([])
plt.xlim(0, 8)
plt.ylabel('Input (mV)')
plt.legend()

V_ax = plt.subplot(212)
plt.plot(mpfc_single_sim.t_vec, mpfc_single_sim.V[:, 0], 'k-', label = 'mPFC')
plt.plot(lhb_single_sim.t_vec, lhb_single_sim.V[:, 0], 'r-', alpha = 0.9, label = 'LHb')
plt.xlim(0, 8)
plt.ylabel('V (mV)')
plt.xlabel('Time ($\\tau_\\mathrm{{mem}}$)')
plt.legend()

plt.tight_layout()

plt.show()

print('mPFC EPSP AUC: {:.2f}mV tau'.format(AUC_mpfc))
print('LHb EPSP AUC: {:.2f}mV tau'.format(AUC_lhb))

#%%

class SynapticBombardment(object):

    def __init__(self, dt = 0.001):
        self.dt = dt

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

    def generate_trigger(self, rate, T, no_neurons, seed = 42):
        """Generate a sparse matrix with ones placed according to a Poisson point process."""

        trigger_mat = np.zeros((int(T/self.dt), no_neurons))
        np.random.seed(seed)
        rands = np.random.uniform(0, 1, trigger_mat.shape)
        trigger_mat[self.poisson_rtp(rate) >= rands] = 1

        self.trigger_mat = trigger_mat

        return trigger_mat

    def conv(self):
        """Convolve trigger matrix with synaptic kernel."""

        I = np.empty_like(self.trigger_mat)

        for i in range(self.trigger_mat.shape[1]):
            I[:, i] = np.convolve(self.trigger_mat[:, i], self.kernel, 'same')

        self.I = I

        return I

#%%

def poisson_rtp(rate, dt):
    """Convert the rate of a Poisson point process to a probability."""
    return 1 - np.exp(-rate * dt)

def generate_poisson_trigger(rate, T, dt, seed = 42):
    """Generate a sparse vector with ones placed according to a Poisson point process.

    Inputs:
        rate -- rate of the poisson process
        T    -- length of the vector in time units
        dt   -- timestep
        seed -- random seed
    """

    trigger_vec = np.zeros(int(T/dt))
    np.random.seed(seed)
    rands = np.random.uniform(0, 1, len(trigger_vec))
    trigger_vec[poisson_rtp(rate, dt) >= rands] = 1

    return trigger_vec

def generate_poisson_trigger_mat(no_neurons, rate, T, dt, seed = 42):
    trigger_mat = np.zeros((int(T/dt), no_neurons))
    np.random.seed(seed)
    rands = np.random.uniform(0, 1, trigger_mat.shape)
    trigger_mat[poisson_rtp(rate, dt) >= rands] = 1

    return trigger_mat

def conv_trigger_mat(trigger_mat, kernel):

    convolved = np.empty_like(trigger_mat)

    for i in range(trigger_mat.shape[1]):
        convolved[:, i] = np.convolve(trigger_mat[:, i], kernel, 'same')

    return convolved

tm = generate_poisson_trigger_mat(200, lhb_rate, T_bombardment, dt)
cm = conv_trigger_mat(tm, mpfc_kernel)

plt.figure()
plt.plot(cm)
plt.show()


T_bombardment = 10 * 1e3/75

lhb_rate = 0.015 * membrane_tau
mpfc_rate = lhb_rate * AUC_lhb / AUC_mpfc

lhb_trigger_vec = generate_poisson_trigger(lhb_rate, T_bombardment, dt)
mpfc_trigger_vec = generate_poisson_trigger(mpfc_rate, T_bombardment, dt)

lhb_vin = np.convolve(lhb_trigger_vec, lhb_kernel, 'same')[:, np.newaxis]
mpfc_vin = np.convolve(mpfc_trigger_vec, mpfc_kernel, 'same')[:, np.newaxis]

print('Performing linear simulations...')
lhb_bbd_lin_sim = IAmod.Simulation(linmod, -60, lhb_vin, dt)
mpfc_bbd_lin_sim = IAmod.Simulation(linmod, -60, mpfc_vin, dt)

print('Performing ia simulations...')
lhb_bbd_ia_sim = IAmod.Simulation(iamod, -60, lhb_vin, dt)
mpfc_bbd_ia_sim = IAmod.Simulation(iamod, -60, mpfc_vin, dt)
print('Done!')

lhb_bbd_ia_sim.simple_plot()
mpfc_bbd_ia_sim.simple_plot()



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
