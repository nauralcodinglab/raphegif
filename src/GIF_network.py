#%% IMPORT MODULES

from __future__ import division

import numpy as np

from src.Tools import timeToIntVec


#%% DEFINE GIF NETWORK CLASS

class GIFnet(object):

    def __init__(self, **kwargs):
        """
        Valid kwargs are:
            ser_mod
            gaba_mod
            propagation_delay
            name
            connectivity_matrix
            gaba_kernel
            dt (default 0.1)
            check_kwargs (set True to turn on input checking for __init__)
        """

        # Defaults for kwargs.
        kwargs['dt'] = kwargs.get('dt', 0.1)
        kwargs['check_kwargs'] = kwargs.get('check_kwargs', False)

        # List of attributes absolutely needed to perform simulation.
        self._necessary_attributes = ['ser_mod', 'gaba_mod',
                                      'propagation_delay',
                                      'connectivity_matrix',
                                      'gaba_kernel', 'dt']

        valid_keys = self._necessary_attributes + ['check_kwargs', 'name']

        # Initialize attributes.
        for key, val in kwargs.iteritems():
            if key not in valid_keys and kwargs['check_kwargs']:
                raise TypeError('Invalid keyword argument {}'.format(key))
            elif key != 'check_kwargs': # Don't save value of check_kwargs
                setattr(self, key, val)
            else:
                pass

    @property
    def no_gaba_neurons(self):
        return self.connectivity_matrix.shape[1]

    @property
    def no_ser_neurons(self):
        return self.connectivity_matrix.shape[0]

    def _check_attributes_inputs(self, ser_input, gaba_input):
        """
        Method to check that all attributes needed to run the
        network simulation have been initialized properly, and that inputs are the right shape.
        Raises an error if not.
        """

        # Check that attributes have been created.
        attrs_not_found = []
        for key in self._necessary_attributes:
            if not hasattr(self, key):
                attrs_not_found.append(key)

        if len(attrs_not_found) > 0:
            raise AttributeError(
                'Uninitialized attributes necessary for simulation.\n'
                'The following must be created: {}'.format(', '.join(attrs_not_found)))

        # Check that input arrays are the right size.
        if not gaba_input.shape[0] == self.no_gaba_neurons:
            raise ValueError('gaba_input nrows and connectivity_matrix ncols do not match.')

        if not ser_input.shape[0] == self.no_ser_neurons:
            raise ValueError('ser_input nrows and connectivity_matrix nrows do not match.')

        # Check timesteps.
        if self.ser_mod.dt != self.dt:
            raise ValueError('Instance dt and ser_mod.dt do not match!')
        if self.gaba_mod.dt != self.dt:
            raise ValueError('Instance dt and gaba_mod.dt do not match!')



    def simulate(self, input_arrs):
        """
        Perform GIF network simulation.

        ser_input = input_arrs['ser_input']
        gaba_input = input_arrs['gaba_input']
        """

        ser_input = input_arrs['ser_input']
        gaba_input = input_arrs['gaba_input']

        # First, check that all necessary attributes have been initialized.
        self._check_attributes_inputs(ser_input, gaba_input)

        # Iterate over all GABA neurons in population.
        gaba_spktimes = []
        gaba_outmat = np.empty((self.no_gaba_neurons, gaba_input.shape[1]), dtype = np.float32) # Spks convolved with IPSC kernel
        for gaba_no in range(self.no_gaba_neurons):
            t, V, eta, v_T, spks = self.gaba_mod.simulate(
                gaba_input[gaba_no, :], self.gaba_mod.El
            )
            gaba_spktimes.append(spks)
            gaba_outmat[gaba_no, :] = np.convolve(
                timeToIntVec(spks, gaba_input.shape[1] * self.dt, self.dt), self.gaba_kernel, 'same'
            ).astype(np.float32)

            # Save a sample trace.
            if gaba_no == 0:
                gaba_ex = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': gaba_input[gaba_no, :]}

            print '\rSimulating GABA neurons {:.1f}%'.format(100 * (gaba_no + 1) / self.no_gaba_neurons),
        print '\n',

        # Add propagation delay to GABA input.
        gaba_outmat = np.roll(gaba_outmat, int(self.propagation_delay / self.dt), axis = 1)
        gaba_outmat[:, :int(self.propagation_delay / self.dt)] = 0

        # Transform GABA output into 5HT input using connectivity matrix.
        gaba_inmat = np.dot(connectivity_matrix, gaba_outmat)

        # Create a dict to hold 5HT example traces.
        ser_examples = {}

        # Allocate lists to hold 5HT spiketrains.
        ser_spktimes = {
            'ib': [],
            'reg': []
        }

        for ser_no in range(self.no_ser_neurons):

            # Try with and without feed-forward inhibition
            for ib_status, ib_multiplier in zip(('ib', 'reg'), (1, 0)):
                I = ser_input[ser_no, :] + gaba_inmat[ser_no, :] * ib_multiplier
                t, V, eta, v_T, spks = self.ser_mod.simulate(
                    I,
                    self.ser_mod.El
                )
                ser_spktimes[ib_status].append(spks)

                # Save a sample trace.
                if ser_no == 0:
                    ser_examples[ib_status] = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

            print '\rSimulating 5HT neurons {:.1f}%'.format(100 * (ser_no + 1) / self.no_ser_neurons),
        print '\n'

        output_dict =  {
            'ser_spktimes': ser_spktimes,
            'ser_examples': ser_examples,
            'gaba_spktimes': gaba_spktimes,
            'gaba_examples': gaba_ex,
            'gaba_inmat': gaba_inmat
        }

        return output_dict


# Simple test for GIFnet
if __name__ == '__main__':

    from src.GIF import GIF

    no_ser_neurons = 5
    no_gaba_neurons = 10
    T = 100
    dt = 0.1

    connectivity_matrix = np.random.uniform(size = (no_ser_neurons, no_gaba_neurons))

    ser_input = 0.1 * np.ones((no_ser_neurons, int(T / dt)), dtype = np.float32)
    gaba_input = 0.1 * np.ones((no_gaba_neurons, int(T / dt)), dtype = np.float32)

    # Try initializing GIFnet.
    test_gifnet = GIFnet(
        ser_mod = GIF(dt),
        gaba_mod = GIF(dt),
        gaba_kernel = [1, 1, 1, 0.5],
        propagation_delay = 1.,
        connectivity_matrix = connectivity_matrix,
        dt = dt
    )

    # Try running a simple simulation.
    test_gifnet.simulate({'ser_input': ser_input, 'gaba_input': gaba_input})


#%% DEFINE OTHER USEFUL TOOLS

def plot_sample(sample):
    """Simple plot of example neuron from GIFnet
    """
    spec = gs.GridSpec(3, 1, height_ratios = [0.2, 1, 0.1])

    plt.figure(figsize = (5, 4))

    plt.subplot(spec[0, :])
    plt.plot(sample['t'], sample['I'], '-', color = 'gray')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.xticks([])
    plt.ylabel('$I$ (nA)')

    plt.subplot(spec[1, :])
    plt.plot(sample['t'], sample['V'], 'k-')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.xticks([])
    plt.ylabel('$V$ (mV)')

    plt.subplot(spec[2, :])
    plt.plot(sample['spks'], np.zeros_like(sample['spks']), 'k|')
    plt.xlim(sample['t'][0], sample['t'][-1])
    plt.ylabel('Repeat no.')

    plt.tight_layout()

def PSTH(spktrain, window_width, no_neurons, dt = 0.1):
    """
    Obtain the population firing rate with a resolution of `window_width`.
    """
    kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
    psth = np.convolve(spktrain, kernel, 'same')
    return psth
