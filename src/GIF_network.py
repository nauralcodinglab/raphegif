#%% IMPORT MODULES

from __future__ import division

import warnings

import numpy as np
import h5py

from src.Tools import timeToIntVec


#%% DEFINE GIF NETWORK CLASS

class GIFnet(object):

    def __init__(self, name = None, dt = 0.1,
                 ser_mod = None, gaba_mod = None,
                 propagation_delay = 0., connectivity_matrix = None,
                 gaba_kernel = None):
        """
        Feed-forward network model with GIF units.

        Model architecture is strictly feed-forward, with a layer of
        GABA neurons feeding onto a layer of 5HT neurons. Both layers
        receive direct external input.

        Attributes:
            name (str)
            dt (float, default 0.1)
                -- Timestep to use for simulation (ms).
            ser_mod (list of GIFs)
                -- List of GIFs to use for 5HT cells.
            gaba_mod (list of GIFs)
                -- List of GIFs to use for GABA cells.
            propagation_delay (float, default 0.)
                -- Synaptic propagation delay from GABA to 5HT layer.
            connectivity_matrix (2D array)
                -- Feed-forward connectivity matrix. Target (5HT) neurons
                are along rows.
            gaba_kernel (1D array)
                -- IPSC kernel to convolve with GABA cell spikes for
                feed-forward inhibition of 5HT cells.
            no_gaba_neurons (int)
                -- Number of gaba neurons in network.
            no_ser_neurons (int)
                -- Number of 5HT neurons in network.

        Important methods:
            clear_interpolated_filters()
                -- Deletes cached interpolated filters in attached GIF models.
                Makes total object size smaller for saving to disk.
            simulate(ser_input, gaba_input, do_feedforward)
                -- Runs network simulation using external input to both populations.
                Optionally, simulates only one layer, or both layers but with
                feedforward connections severed.
        """

        self.name = name
        self.dt = dt

        self.ser_mod = ser_mod
        self.gaba_mod = gaba_mod
        self.propagation_delay = propagation_delay
        self.connectivity_matrix = connectivity_matrix
        self.gaba_kernel = gaba_kernel

    @property
    def no_gaba_neurons(self):
        if self.gaba_mod is not None:
            return len(self.gaba_mod)
        else:
            return 0

    @property
    def no_ser_neurons(self):
        if self.ser_mod is not None:
            return len(self.ser_mod)
        else:
            return 0

    def clear_interpolated_filters(self):
        """Interpolated eta and gamma filters in the ser and gaba mods can be
        large and expensive to store. This removes the cached interpolated
        versions to cut down the disk size of the GIFnet.
        """

        for i in range(len(self.ser_mod)):
            self.ser_mod[i].eta.clearInterpolatedFilter()
            self.ser_mod[i].gamma.clearInterpolatedFilter()

        for i in range(len(self.gaba_mod)):
            self.gaba_mod[i].eta.clearInterpolatedFilter()
            self.gaba_mod[i].gamma.clearInterpolatedFilter()

    def _simulate_gaba(self, gaba_input, verbose = False):
        """Simulate response of GABA neurons to gaba_input.

        Input:
            gaba_input (2D array)
                -- Input to GABA cells. Must have no_gaba_neurons rows and T/dt columns.
            verbose (bool, default False)
                -- Print information about progress.

        Returns:
            Dict with spktimes (list of lists of spktimes from each cell)
            and example (sample trace).
        """

        # Input checks.
        if gaba_input.shape[0] != self.no_gaba_neurons:
            raise ValueError(
                'gaba_input first axis length {} and no. gaba_mods {} do not match'.format(
                    gaba_input.shape[0], len(self.gaba_mod)
                )
            )

        gaba_spktimes = []
        for gaba_no in range(self.no_gaba_neurons):
            if verbose:
                print('Simulating gaba neurons {:.1f}%'.format(100 * (gaba_no + 1)/self.no_gaba_neurons))
            t, V, eta, v_T, spks = self.gaba_mod[gaba_no].simulate(
                gaba_input[gaba_no, :], self.gaba_mod[gaba_no].El
            )
            gaba_spktimes.append(spks)

            # Save a sample trace.
            if gaba_no == 0:
                gaba_example = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': gaba_input[gaba_no, :]}

        return {'spktimes': gaba_spktimes, 'example': gaba_example}

    def _convolve_feedforward(self, gaba_spktimes, T):
        """Convert GABA spks into feedforward input to 5HT cells.

        Inputs:
            gaba_spktimes (list of lists)
                -- Spiketimes of GABA cells in network.
            T (float)
                -- Duration of simulation (ms).

        Returns:
            2D array of shape (no_ser_neurons, T/dt) with synaptic input to 5HT
            neurons computed based on instance gaba_kerneli, propagation_delay,
            and connectivity_matrix.
        """

        # Input checks.
        if len(gaba_spktimes) != self.no_gaba_neurons:
            raise ValueError(
                'Number of gaba_neurons for which spktimes are given ({}) '
                'must match no_gaba_neurons in GIFnet ({})'.format(
                    len(gaba_spktimes), self.no_gaba_neurons
                )
            )
        elif self.connectivity_matrix is None:
            raise AttributeError(
                'Feed-forward inputs cannot be computed if GIFnet connectivity_matrix '
                'is not set.'
            )
        elif len(gaba_spktimes) != self.connectivity_matrix.shape[1]:
            raise ValueError(
                'No. of gaba_spktimes ({}) and connectivity matrix no cols ({}) '
                'must match.'.format(len(gaba_spktimes), self.connectivity_matrix.shape[1])
            )

        # Convert gaba_spktimes to spktrains.
        gaba_spktrains = np.empty((self.no_gaba_neurons, int(T / self.dt + 0.5)), dtype = np.float32)
        for gaba_no in range(self.no_gaba_neurons):
            gaba_spktrains[gaba_no, :] = np.convolve(
                timeToIntVec(
                    gaba_spktimes[gaba_no], T, self.dt
                ), self.gaba_kernel, 'same'
            ).astype(np.float32)

        # Add propagation delay to GABA input.
        gaba_spktrains = np.roll(gaba_spktrains, int(self.propagation_delay / self.dt), axis = 1)
        gaba_spktrains[:, :int(self.propagation_delay / self.dt)] = 0

        # Transform GABA output into 5HT input using connectivity matrix.
        feedforward_input = np.dot(self.connectivity_matrix, gaba_spktrains)

        return feedforward_input

    def _simulate_ser(self, ser_input, feedforward_input = None, verbose = False):
        """Simulate response of 5HT neurons to ser_input.

        Input:
            ser_input (2D array)
                -- External input to 5HT cells. Must have no_ser_neurons rows and T/dt columns.
            feedforward_input (2D array)
                -- Feedforward input to 5HT cells from GABA population.
                Must have same shape as ser_input.
            verbose (bool, default False)
                -- Print information about progress.

        Returns:
            Dict with spktimes (list of lists of spktimes from each cell)
            and example (sample trace).
        """

        # Check that ser_input has as many rows as we have models.
        if ser_input.shape[0] != self.no_ser_neurons:
            raise ValueError(
                'ser_input first axis length {} and no. ser_mods {} do not match'.format(
                    ser_input.shape[0], len(self.ser_mod)
                )
            )

        # Check that input shapes match.
        if feedforward_input is not None:
            if feedforward_input.shape == ser_input.shape:
                I = ser_input + feedforward_input
            else:
                raise ValueError(
                    'Shape of ser_input {} does not match shape of feedforward_input {}.'.format(
                        ser_input.shape, feedforward_input.shape
                    )
                )
        else:
            I = ser_input

        # Loop over cells.
        ser_spktimes = []
        for ser_no in range(self.no_ser_neurons):
            if verbose:
                print('Simulating ser neurons {:.1f}%'.format(100 * (ser_no + 1)/self.no_ser_neurons))
            t, V, eta, v_T, spks = self.ser_mod[ser_no].simulate(
                I[ser_no, :], self.ser_mod[ser_no].El
            )
            ser_spktimes.append(spks)

            # Save a sample trace.
            if ser_no == 0:
                ser_example = {'t': t, 'V': V, 'eta': eta, 'v_T': v_T, 'spks': spks, 'I': I}

        return {'spktimes': ser_spktimes, 'example': ser_example}

    def simulate(self, ser_input = None, gaba_input = None, do_feedforward = True, verbose = True):
        """
        Perform GIF network simulation.

        Inputs:
            ser_input (2D array or None)
                -- External input for driving 5HT neurons in the network.
                Should have shape (no_ser_neurons, timesteps).
                Set to None to skip simulating 5HT cells.
            gaba_input (2D array or None)
                -- External input for driving GABA neurons in the network.
                Should have shape (no_gaba_neurons, timesteps).
                Set to None to skip simulating GABA cells.
            do_feedforward (bool, default True)
                -- Connect GABA and 5HT layers. If false, responses of both
                populations to external input are simulated, but GABA output
                is not given to 5HT neurons.

        Returns:
            GIFnetSim object with simulation results.
        """

        # Input checks.
        if ser_input is None and gaba_input is None:
            raise ValueError(
                'One of ser_input or gaba_input must not be None.'
            )
        elif ser_input is not None and gaba_input is not None and ser_input.shape[1] != gaba_input.shape[1]:
            raise ValueError(
                'ser_input no_timesteps ({}) must match gaba_input no_timesteps ({})'.format(
                    ser_input.shape[1], gaba_input.shape[1]
                )
            )
        elif ser_input is not None and ser_input.shape[0] != self.no_ser_neurons:
            raise ValueError(
                'ser_input no. rows ({}) must match GIFnet.no_ser_neurons ({})'.format(
                    ser_input.shape[0], self.no_ser_neurons
                )
            )
        elif gaba_input is not None and gaba_input.shape[0] != self.no_gaba_neurons:
            raise ValueError(
                'gaba_input no. rows ({}) must match GIFnet.no_gaba_neurons ({})'.format(
                    gaba_input.shape[0], self.no_gaba_neurons
                )
            )

        # Initialize GIFnetSim to hold output.
        if ser_input is not None:
            T = ser_input.shape[1] * self.dt
        else:
            T = gaba_input.shape[1] * self.dt
        output = GIFnetSim(T, self.dt, name = getattr(self, 'name', None))

        # Simulate GABA neurons (if applicable).
        if gaba_input is not None:
            gabasim = self._simulate_gaba(gaba_input, verbose)
            output.no_gaba_neurons = self.no_gaba_neurons
            output.gaba_example = gabasim['example']
            output.gaba_spktimes = gabasim['spktimes']
            del gabasim

        # Simulate 5HT neurons (if applicable).
        if ser_input is not None:

            # Generate GABA input to 5HT cells based on GABA spks.
            if gaba_input is not None and do_feedforward:
                feedforward_input = self._convolve_feedforward(output.gaba_spktimes, T)

                # Save output.
                output.connectivity_matrix = self.connectivity_matrix
                output.propagation_delay = self.propagation_delay
                output.feedforward_input = feedforward_input

                output.metadata['feedforward_input'] = 'True'
            else:
                feedforward_input = None
                output.metadata['feedforward_input'] = 'False'

            # Simulate 5HT neurons.
            sersim = self._simulate_ser(ser_input, feedforward_input, verbose)
            output.no_ser_neurons = self.no_ser_neurons
            output.ser_example = sersim['example']
            output.ser_spktimes = sersim['spktimes']
            del sersim

        return output

class GIFnetSim(object):

    def __init__(self, T, dt, name = None,
                 no_ser_neurons = 0, no_gaba_neurons = 0,
                 ser_spktimes = None, ser_example = None,
                 gaba_spktimes = None, gaba_example = None,
                 feedforward_input = None, connectivity_matrix = None,
                 propagation_delay = None,
                 metadata = {}):
        """GIFnetSim class for holding GIFnet simulations

        Main feature is to save output of GIFnet simulations in HDF5 format.
        """

        self.name = name
        self.metadata = metadata

        self.T = T
        self.dt = dt

        self.no_ser_neurons = no_ser_neurons
        self.no_gaba_neurons = no_gaba_neurons

        self.ser_spktimes = ser_spktimes
        self.gaba_spktimes = gaba_spktimes
        self.ser_example = ser_example
        self.gaba_example = gaba_example

        self.feedforward_input = feedforward_input
        self.connectivity_matrix = connectivity_matrix
        self.propagation_delay = propagation_delay

    def set_metadata(self, metadata):
        self.metadata = metadata

    @property
    def T_ind(self):
        return int(self.T/self.dt + 0.5)

    def save_hdf(self, fname, compression_level = 5):
        """Save GIFnetSim in hierarchical data format (HDF5)

        Inputs:
            fname (str)
                -- Path to save HDF file.
            compression_level (int 0-9)
                -- Amount of gzip compression for spiketrains.
        """

        f = h5py.File(fname, 'w')

        ### Save metadata.
        meta_attrs = ['name', 'T', 'dt', 'no_ser_neurons',
                      'no_gaba_neurons', 'propagation_delay']
        for attr_ in meta_attrs:
            if getattr(self, attr_, None) is not None:
                f.attrs[attr_] = getattr(self, attr_)
            else:
                f.attrs[attr_] = ''

        if len(self.metadata) > 0:
            for key, val in self.metadata.iteritems():
                f.attrs[key] = val

        ### Save 5HT data.
        if self.no_ser_neurons > 0:
            sergroup = f.create_group('ser')

            # Save ser_example.
            if self.ser_example is not None:
                serex = sergroup.create_group('example')

                for key in ['t', 'V', 'eta', 'v_T', 'I']:
                    serex.create_dataset(
                        key, data = self.ser_example[key],
                        dtype = np.float32
                    )

                tmp_spktrain = timeToIntVec(
                    self.ser_example['spks'], self.T, self.dt
                )
                assert len(tmp_spktrain) == len(self.ser_example['t']), 'tmp_spktrain length {} and ser_example[\'t\'] length {} not equal'.format(
                        len(tmp_spktrain), len(self.ser_example['t'])
                    )
                serex.create_dataset(
                    'spks', data = tmp_spktrain,
                    dtype = np.int8, compression = compression_level
                )

                del tmp_spktrain

            # Save ser_spktimes.
            if self.ser_spktimes is not None:
                sergroup.create_dataset(
                    'spktrains',
                    shape = (self.no_ser_neurons, self.T_ind),
                    dtype = np.int8, compression = 5
                )
                for i, spktimes in enumerate(self.ser_spktimes):
                    sergroup['spktrains'][i, :] = timeToIntVec(
                        spktimes, self.T, self.dt
                    )

        # Catch whether any initialized attrs are not being saved because no_ser_neurons == 0.
        elif not all([getattr(self, x) is None for x in ['ser_example', 'ser_spktimes']]):
            warnings.warn(
                '5HT data not being saved because GIFnetSim.no_ser_neurons == 0, '
                'even though some 5HT-related attributes have been set.',
                UserWarning
            )

        ### Save GABA data.
        if self.no_gaba_neurons > 0:
            gabagroup = f.create_group('gaba')

            # Save gaba_example.
            if self.gaba_example is not None:
                gabaex = gabagroup.create_group('example')

                for key in ['t', 'V', 'eta', 'v_T', 'I']:
                    gabaex.create_dataset(
                        key, data = self.gaba_example[key],
                        dtype = np.float32
                    )

                tmp_spktrain = timeToIntVec(
                    self.gaba_example['spks'], self.T, self.dt
                )
                assert len(tmp_spktrain) == len(self.gaba_example['t'])
                gabaex.create_dataset(
                    'spks', data = tmp_spktrain,
                    dtype = np.int8, compression = compression_level
                )

                del tmp_spktrain

            # Save gaba_spktimes.
            if self.gaba_spktimes is not None:
                gabagroup.create_dataset(
                    'spktrains',
                    shape = (self.no_gaba_neurons, self.T_ind),
                    dtype = np.int8, compression = compression_level
                )
                for i, spktimes in enumerate(self.gaba_spktimes):
                    gabagroup['spktrains'][i, :] = timeToIntVec(
                        spktimes, self.T, self.dt
                    )

            # Save gaba input to ser cells.
            if self.feedforward_input is not None:
                sergroup.create_dataset(
                    'feedforward_input',
                    data = self.feedforward_input,
                    dtype = np.float32
                )

            # Save connectivity matrix.
            if self.connectivity_matrix is not None:
                f.create_dataset(
                    'connectivity_matrix',
                    data = self.connectivity_matrix, dtype = np.int8,
                    compression = compression_level
                )

        elif not all([getattr(self, x) is None for x in ['gaba_example', 'gaba_spktimes']]):
            warnings.warn(
                'GABA data not being saved because GIFnetSim.no_gaba_neurons == 0, '
                'even though some GABA-related attributes have been set.',
                UserWarning
            )

        ### Close file.
        f.close()


# Simple test for GIFnet
if __name__ == '__main__':

    import os

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
        ser_mod = [GIF(dt)] * no_ser_neurons,
        gaba_mod = [GIF(dt)] * no_gaba_neurons,
        gaba_kernel = [1, 1, 1, 0.5],
        propagation_delay = 1.,
        connectivity_matrix = connectivity_matrix,
        dt = dt
    )

    # Try running a simple simulation.
    test_gifnetsim = test_gifnet.simulate(ser_input = ser_input, gaba_input = gaba_input)

    # Try saving output.
    testfilename = 'testgifnetsimfile.hdf5.test'
    test_gifnetsim.save_hdf(testfilename)
    os.remove(testfilename)


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
