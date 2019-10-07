#%% IMPORT MODULES

from __future__ import division
import os
import multiprocessing as mp
import itertools
import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy as sp

from grr.SubthreshGIF import SubthreshGIF_K
from grr import Tools


#%% DEFINE MEMBRANE FILTER COMPARATOR CLASS

def _PSDworker(args):

    """
    Worker function to extract PSD of V and I arrays.
    Used for parallelization with multiprocessing.

    Args should be a tuple of V_arr, I_arr, and dt.

    N.B. Unfortunately, in order to get multiprocessing to work, this function has to be defined at the top level of the module (rather than as a KappaComparator staticmethod).
    """

    V_arr = args[0]
    I_arr = args[1]
    dt = args[2]
    nperseg = 1e5

    I_f_arr = []
    I_PSD_arr = []
    V_f_arr = []
    V_PSD_arr = []

    # Get PSD
    for j in range(V_arr.shape[1]):

        I_f, I_PSD = sp.signal.welch(I_arr[:, j], 1000. / dt, 'hann', nperseg)
        V_f, V_PSD = sp.signal.welch(V_arr[:, j], 1000. / dt, 'hann', nperseg)

        I_f_arr.append(I_f)
        I_PSD_arr.append(I_PSD)
        V_f_arr.append(V_f)
        V_PSD_arr.append(V_PSD)

    # Convert PSD lists to arrays.
    I_f_arr = np.array(I_f_arr).T
    I_PSD_arr = np.array(I_PSD_arr).T
    V_f_arr = np.array(V_f_arr).T
    V_PSD_arr = np.array(V_PSD_arr).T

    return (I_f_arr, I_PSD_arr, V_f_arr, V_PSD_arr)


class KappaComparator(object):

    def __init__(self, title, models):

        if not all([models[0].dt == mod.dt for mod in models]):
            raise ValueError('All models should have same dt.')

        self.title = title

        # Models from which to extract Kappa
        self.models = models
        self.dt = models[0].dt

        # Input noise parameters
        self._noiseParamsSet = False

        # Simulated data
        self._simulatedVoltage = False

        # Membrane filter properties.
        self._kappaExtracted = False


    def setNoiseParams(self, sigma, tau, duration):

        # Check whether noise params were already set.
        if self._noiseParamsSet:
            if raw_input('Noise params already set. Overwrite? (y/n)').lower() == 'y':
                print 'Overwriting noise params.'
                pass
            else:
                print 'Cancelled reseting noise_params.'
                return

        # Check for correct input.
        input_lens = []
        for input_ in [sigma, tau, duration]:

            try:
                input_lens.append(len(input_))
            except TypeError:
                pass

        if len(input_lens) > 0:
            if any([l != len(self.models) for l in input_lens]):
                raise TypeError('Need to specify sigma, tau, and duration '
                'values for each model, or single value to use for all models')

        # Set params.
        self.noiseSigma = sigma
        self.noiseTau = tau
        self.noiseDuration = duration

        # Set flag.
        self._noiseParamsSet = True


    def simulateNoiseResponse(self, voltages, verbose = True):

        """
        Simulate voltage response of models to noisy input around specified mean voltages.
        """

        # Check whether V-response was already simulated.
        if self._simulatedVoltage:
            if raw_input('Voltage response already simulated. Redo? (y/n)').lower() == 'y':
                print 'Redoing voltage response simulation.'
                pass
            else:
                print 'Cancelling simulation of voltage response.'
                return

        # Convert voltages to an iterable if it isn't already one.
        try:
            iter(voltages)
        except TypeError:
            voltages = [voltages]

        # Initialize class attributes.
        self._noiseVoltageOffsets = voltages
        self._simulatedVoltageResponse = []
        self._simulatedInjectedCurrent = []

        # Generate base noise.
        if verbose: print 'Generating base noise.'
        base_noise = Tools.generateOUprocess(self.noiseDuration, self.noiseTau,
        0., self.noiseSigma, self.dt)
        base_noise = base_noise.astype(np.float32)

        # Iterate over models.
        for i in range(len(self.models)):

            mod = self.models[i]

            # Generate input current array for this model.
            I_arr = np.empty((int(self.noiseDuration/self.dt), len(voltages)),
            dtype = np.float32)
            for j in range(len(voltages)):
                if verbose:
                    print '\rOffsetting noise model {}: {:0.1f}%'.format(i,
                    100. * (j+1) / I_arr.shape[1]),
                I_offset = mod.simulateVClamp(1, voltages[j], None)[1].mean()
                I_arr[:, j] = base_noise + I_offset.astype(np.float32)

            # Generate voltage response array for this model.
            if verbose: print ''
            V_arr = np.empty_like(I_arr)
            for j in range(I_arr.shape[1]):
                if verbose:
                    print '\rSimulating voltage response model {}: {:0.1f}%'.format(
                    i, 100. * (j+1) / I_arr.shape[1]),
                V_arr[:, j] = mod.simulate(I_arr[:, j], voltages[j])[1].astype(np.float32)
            if verbose: print ''

            # Store output in class attributes.
            self._simulatedVoltageResponse.append(V_arr)
            self._simulatedInjectedCurrent.append(I_arr)

        # Set flag.
        self._simulatedVoltage = True



    def extractKappa(self, verbose = True):

        """
        Extract membrane filters from models.
        """

        if self._kappaExtracted:
            if raw_input('Kappa already extracted. Redo? (y/n)').lower() == 'y':
                print 'Redoing kappa extraction.'
                pass
            else:
                print 'Kappa extraction cancelled.'
                return

        # Initialize attributes to hold output.
        self._I_f = []
        self._I_PSD = []
        self._V_f = []
        self._V_PSD = []
        self._kappa_f = []
        self._kappa_impedance = []

        # Make iterable to pass to KappaComarator._PSDworker()
        noise_input_iter = itertools.izip(self._simulatedVoltageResponse,
        self._simulatedInjectedCurrent,
        [self.dt for i in range(len(self.models))])

        # Extract filters.
        if __name__ == '__main__':

            if verbose: print 'Extracting filters in parallel.'

            pool_ = mp.Pool()

            for out in pool_.map(_PSDworker, noise_input_iter):

                self._I_f.append(out[0])
                self._I_PSD.append(out[1])
                self._V_f.append(out[2])
                self._V_PSD.append(out[3])

                self._kappa_f.append(out[0])
                self._kappa_impedance.append(np.sqrt(out[3]/out[1]))

            pool_.close()
            pool_.join()

        else:

            if verbose:
                print 'Cannot extract filters in parallel outside of main.'
                print 'Iterating normally instead.'

            for out in itertools.imap(_PSDworker, noise_input_iter):

                self._I_f.append(out[0])
                self._I_PSD.append(out[1])
                self._V_f.append(out[2])
                self._V_PSD.append(out[3])

                self._kappa_f.append(out[0])
                self._kappa_impedance.append(np.sqrt(out[3]/out[1]))

        if verbose: print 'Done!'

        # Set flag.
        self._kappaExtracted = True


    def pickleKappa(self, fname, verbose = True):

        """
        Save the extracted membrane filter.
        """

        # Check whether a filter has been extracted.
        if not self._kappaExtracted:
            raise RuntimeError('No filter to pickle!')

        # Add the extension if necessary.
        if fname[-4:].lower() != '.pyc':
            fname += '.pyc'

        # Check whether a file with the same name already exists.
        if os.path.isfile(fname):

            if raw_input('{} already exists. Overwrite? (y/n)'.format(fname)).lower() == 'y':
                print 'Overwrite confirmed.'
                pass
            else:
                print 'Pickling cancelled.'
                return

        else: pass

        # Create temporary object with just the filter.
        tmp = {
        'kappa_f': self._kappa_f,
        'kappa_impedance': self._kappa_impedance,
        'noise_voltage_offsets': self._noiseVoltageOffsets
        }

        # Pickle the temporary object.
        with open(fname, 'wb') as f:
            if verbose: print 'Pickling kappa...'
            pickle.dump(tmp, f)
            if verbose: print 'Done!'


    def loadKappa(self, fname, verbose = True):

        """
        Load a membrane filter from a .pyc file.
        """

        if fname[-4:].lower() != '.pyc':
            raise ValueError('Only .pyc files can be used.')

        # Warn the user if they are about to erase an extracted filter.
        if self._kappaExtracted:

            if raw_input('Kappa already extracted. Overwrite? (y/n)').lower() == 'y':
                print 'Overwrite confirmed.'
                pass
            else:
                print 'Load cancelled.'
                return

        # Load the filter.
        with open(fname, 'rb') as f:
            if verbose: print 'Loading kappa...'
            tmp = pickle.load(f)
            if verbose: print 'Done!'

        self._kappa_f = tmp['kappa_f']
        self._kappa_impedance = tmp['kappa_impedance']
        self._noiseVoltageOffsets = tmp['noise_voltage_offsets']


    def plotKappa3D(self, model_nos, save_path = None, freq_cutoff = 1e2, reference_C = None, reference_tau = None, verbose = True):

        """
        Make a 3D plot of the membrane filter as a function of voltage and f.

        If save_path is not None, each fig will be saved to the specified location.
        N.B.: save_path = 'path/to/dir/ex' will result in figs named `ex0.png`, `ex1.png`, etc. in `path/to/dir`

        model_nos can be set to 'all' to make plots for all models, but this might take a long time.
        Set freq_cutoff to `None` to plot all frequencies.
        """

        # Input checks.
        if model_nos == 'all':
            if verbose: print 'Making plots of all models.'
            model_nos = [i for i in range(len(self.models))]
        else:
            try: iter(model_nos)
            except TypeError: model_nos = [model_nos]

        try:
            assert len(reference_C) == len(reference_tau) and len(reference_C) == len(model_nos)
        except AssertionError:
            raise TypeError('reference_C and reference_tau must either be '
            'iterables of the same length as the no. of models to plot, or both be None')
        except TypeError:
            if reference_C is not None or reference_tau is not None:
                raise TypeError('reference_C and reference_tau must either be '
                'iterables of the same length as the no. of models to plot, or both be None')

        # Iterate over models.
        for mod_no in model_nos:

            # Create temporary arrays to use for 3D plot.
            F = self._kappa_f[mod_no]
            V = np.broadcast_to(self._noiseVoltageOffsets[np.newaxis, :], F.shape)
            I = self._kappa_impedance[mod_no] * 1e-3

            # Subset arrays based on frequency.
            if freq_cutoff is not None:
                freq_sub = np.min(np.where(F[:, 0] >= freq_cutoff)[0])
                F = F[:freq_sub, :]
                V = V[:freq_sub, :]
                I = I[:freq_sub, :]

            F = np.log10(F)

            # Make figure.
            fig3d = plt.figure(figsize = (12, 5))

            fig3d.suptitle('Model {}'.format(mod_no))

            ax0 = plt.subplot2grid((2, 5), (0, 0), projection = '3d', colspan = 2, rowspan = 2)
            ax0.set_title('$\kappa$')
            ax0.plot_surface(F, V, I, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
            ax0.set_ylim3d(ax0.get_ylim3d()[1], ax0.get_ylim3d()[0])
            ax0.set_xticks([-1, 0, 1, 2])
            ax0.set_xticklabels(['$10^{{{}}}$'.format(tick) for tick in ax0.get_xticks()])
            ax0.set_yticks([-45, -55, -65, -75])
            ax0.set_xlabel('$f$ (Hz)')
            ax0.set_ylabel('Vm (mV)')
            ax0.set_zlabel('Impedance (G$\Omega$)')

            ax1 = plt.subplot2grid((2, 5), (0, 2), projection = '3d', colspan = 2, rowspan = 2)
            ax1.set_title('Perithreshold $\kappa$')
            ax1.plot_surface(F[:, -15:], V[:, -15:], I[:, -15:], rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
            ax1.set_ylim3d(ax1.get_ylim3d()[1], ax1.get_ylim3d()[0])
            ax1.set_xticks([-1, 0, 1, 2])
            ax1.set_xticklabels(['$10^{{{}}}$'.format(tick) for tick in ax0.get_xticks()])
            ax1.set_yticks([-40, -45, -50, -55, -60])
            ax1.set_xlabel('$f$ (Hz)')
            ax1.set_ylabel('Vm (mV)')
            ax1.set_zlabel('Impedance (G$\Omega$)')

            ax2 = plt.subplot2grid((2, 5), (0, 4))
            cond_vals = np.array([self.models[mod_no].gl, self.models[mod_no].gbar_K1, self.models[mod_no].gbar_K2])
            ax2.bar([0, 1, 2], cond_vals * 1e3, color = (0.9, 0.1, 0.1))
            ax2.set_xticks([0, 1, 2])
            ax2.set_xticklabels(['$g_l$', '$g_{k1}$', '$g_{k2}$'])
            ax2.set_ylabel('Conductance (pS)')

            ax3 = plt.subplot2grid((2, 5), (1, 4))
            ax3.plot(1./self.models[mod_no].gl * self.models[mod_no].C, self.models[mod_no].C * 1e3, 'o', label = 'base + k', color = (0.9, 0.1, 0.1))
            if reference_C is not None: ax3.plot(reference_tau[mod_no], reference_C[mod_no], 'o', label = 'base', color = 'gray')
            ax3.set_xlim(0, ax3.get_xlim()[1] * 1.1)
            ax3.set_ylim(0, ax3.get_ylim()[1] * 1.1)
            ax3.legend()
            ax3.set_ylabel('Capacitance (pF)')
            ax3.set_xlabel('$\\tau_m$ (ms)')

            fig3d.subplots_adjust(top = 0.9, left = 0.017, right = 0.96, bottom = 0.1, hspace = 0.25, wspace = 1)

            if save_path is not None:
                fig3d.savefig(save_path + self.title + '{}.png'.format(mod_no), dpi = 600)


#%% INITIALIZE KGIFS

base_model = SubthreshGIF_K()

base_model.C = 0.100 # pF
base_model.gl = 0.001 # nS
base_model.gbar_K1 = 0.
base_model.gbar_K2 = 0.

base_model.m_A = 1.61
base_model.m_Vhalf = -27
base_model.m_k = 0.113
base_model.m_tau = 1.

base_model.h_A = 1.03
base_model.h_Vhalf = -59.2
base_model.h_k = -0.165
base_model.h_tau = 50.

base_model.n_A = 1.55
base_model.n_Vhalf = -16.9
base_model.n_k = 0.114
base_model.n_tau = 100.

base_model.E_K = -101.

gk1_models = [deepcopy(base_model) for i in range(4)]
gk1_models[1].gbar_K1 = -0.005
gk1_models[2].gbar_K1 = 0.010
gk1_models[3].gbar_K1 = 0.020

gk2_models = [deepcopy(base_model) for i in range(3)]
gk2_models[1].gbar_K2 = 0.010
gk2_models[2].gbar_K2 = 0.015

complex_models = [deepcopy(base_model) for i in range(3)]
complex_models[1].gbar_K1 = -0.005
complex_models[1].gbar_K2 = 0.010
complex_models[2].gbar_K1 = 0.010
complex_models[2].gbar_K2 = 0.010


#%% EXTRACT DETAILED POWER SPECTRUM DENSITY

gk1_comparator = KappaComparator('gk1_comparator', gk1_models)
gk1_comparator.setNoiseParams(0.002, 5., 2e5)
gk1_comparator.simulateNoiseResponse(np.linspace(-80, -40, 30))
gk1_comparator.extractKappa()
gk1_comparator.plotKappa3D('all', save_path = '/Users/eharkin/Desktop/')

#%%
gk2_comparator = KappaComparator('gk2_comparator', gk2_models)
gk2_comparator.setNoiseParams(0.002, 5., 2e5)
gk2_comparator.simulateNoiseResponse(np.linspace(-80, -40, 30))
gk2_comparator.extractKappa()
gk2_comparator.plotKappa3D('all', save_path = '/Users/eharkin/Desktop/')

#%%
complex_comparator = KappaComparator('complex_model_comparator', complex_models)
complex_comparator.setNoiseParams(0.002, 5., 2e5)
complex_comparator.simulateNoiseResponse(np.linspace(-80, -40, 30))
complex_comparator.extractKappa()
complex_comparator.plotKappa3D('all', save_path = '/Users/eharkin/Desktop/')

#%%

plt.figure()
plt.plot(complex_comparator._simulatedVoltageResponse[2], alpha = 0.5)
plt.show()
