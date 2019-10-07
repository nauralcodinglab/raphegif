#%% IMPORT MODULES

from __future__ import division

import numpy as np

from .Tools import timeToIntVec
from .Simulation import GIFnet_Simulation


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

        if self.ser_mod is not None:
            for i in range(len(self.ser_mod)):
                self.ser_mod[i].eta.clearInterpolatedFilter()
                self.ser_mod[i].gamma.clearInterpolatedFilter()

        if self.gaba_mod is not None:
            for i in range(len(self.gaba_mod)):
                self.gaba_mod[i].eta.clearInterpolatedFilter()
                self.gaba_mod[i].gamma.clearInterpolatedFilter()

    def _simulate_gaba(self, out, gaba_input, verbose = False):
        """Simulate response of GABA neurons to gaba_input.

        Input:
            out (GIFnet_Simulation)
                -- Directly place simulation results in 'out'.
            gaba_input (3D array)
                -- Input to GABA cells. Must have dimensionality
                [no_sweeps, no_gaba_neurons, timesteps].
            verbose (bool, default False)
                -- Print information about progress.

        Returns:
            Nested list of spike times for internal use.
            All other output is placed in 'out'.
        """

        # Input checks.
        if gaba_input.shape[1] != self.no_gaba_neurons:
            raise ValueError(
                'gaba_input second axis length {} and no. gaba_mods '
                '{} do not match'.format(
                    gaba_input.shape[1], len(self.gaba_mod)
                )
            )

        gaba_spktimes = []
        for sweep_no in range(gaba_input.shape[0]):
            gaba_spktimes_singlesweep = []
            for gaba_no in range(self.no_gaba_neurons):
                if verbose:
                    print(
                        'Simulating gaba neurons sweep {} of {} '
                        '{:.1f}%'.format(
                            sweep_no, gaba_input.shape[0],
                            100 * (gaba_no + 1) / self.no_gaba_neurons
                        )
                    )

                # Simulate cell for this sweep.
                t, V, eta, v_T, spks = self.gaba_mod[gaba_no].simulate(
                    gaba_input[sweep_no, gaba_no, :],
                    self.gaba_mod[gaba_no].El
                )

                # Save spktimes/spktrains.
                gaba_spktimes_singlesweep.append(spks)
                out.gaba_spktrains[sweep_no, gaba_no, :] = timeToIntVec(
                    spks, out.get_T(), self.dt
                )

                # Save sample traces.
                if gaba_no < out.get_no_gaba_examples():
                    tmp_results = {
                        't': t, 'V': V, 'eta': eta,
                        'v_T': v_T, 'I': gaba_input[sweep_no, gaba_no, :]
                    }


                    # Only save pre-initialized channels.
                    for key in out.gaba_examples.keys():
                        out.gaba_examples[key][sweep_no, gaba_no, :] = tmp_results[key]

            gaba_spktimes.append(gaba_spktimes_singlesweep)

        return gaba_spktimes

    def _convolve_feedforward(self, out, gaba_spktimes):
        """Convert GABA spks into feedforward input to 5HT cells.

        Inputs:
            out (GIFnet_Simulation)
                -- Object in which to directly place sample synaptic
                input trains.
            gaba_spktimes (nested list)
                -- Spiketimes of GABA cells in network.
                Must be nested list of depth 3 ([sweep][cell][timestep]).

        Returns:
            3D array of shape (no_sweeps, no_ser_neurons, T/dt) with synaptic
            input to 5HT neurons computed based on instance gaba_kernel,
            propagation_delay, and connectivity_matrix.
        """

        # Input checks.
        if len(gaba_spktimes[0]) != self.no_gaba_neurons:
            # Only checks whether correct no neurons are in first sweep.
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
        elif len(gaba_spktimes[0]) != self.connectivity_matrix.shape[1]:
            # Only checks no neurons in first sweep.
            raise ValueError(
                'No. cells in gaba_spktimes ({}) and connectivity matrix no cols ({}) '
                'must match.'.format(len(gaba_spktimes[0]), self.connectivity_matrix.shape[1])
            )

        # Convert gaba_spktimes to convolved spktrains.
        gaba_conv_spks = np.empty(
            (len(gaba_spktimes),
             self.no_gaba_neurons,
             int(out.get_T() / self.dt + 0.5)),
            dtype = np.float32
        )
        for sweep_no in range(len(gaba_spktimes)):
            for gaba_no in range(self.no_gaba_neurons):
                gaba_conv_spks[sweep_no, gaba_no, :] = np.convolve(
                    timeToIntVec(
                        gaba_spktimes[sweep_no][gaba_no], out.get_T(), self.dt
                    ),
                    self.gaba_kernel, 'same'
                ).astype(np.float32)

        # Add propagation delay to GABA input.
        gaba_conv_spks = np.roll(
            gaba_conv_spks, int(self.propagation_delay / self.dt),
            axis = 2
        )
        gaba_conv_spks[:, :, :int(self.propagation_delay / self.dt)] = 0

        # Transform GABA output tensor into 5HT input using connectivity matrix.
        feedforward_input = np.moveaxis(
            np.tensordot(
                self.connectivity_matrix,
                np.moveaxis(gaba_conv_spks, 0, -1),
                axes = 1
            ),
            -1, 0
        )

        # Save synaptic input trains in 'out'.
        out.ser_examples['feedforward_input'][...] = (
            feedforward_input[:, :out.get_no_ser_examples(), :]
        )

        return feedforward_input

    def _simulate_ser(self, out, ser_input, feedforward_input = None, verbose = False):
        """Simulate response of 5HT neurons to ser_input.

        Input:
            out (GIFnet_Simulation)
                -- Object in which to store simulation results directly.
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
        if ser_input.shape[1] != self.no_ser_neurons:
            raise ValueError(
                'ser_input second axis length {} and no. ser_mods {} do not match'.format(
                    ser_input.shape[1], len(self.ser_mod)
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

        # Loop over sweeps and cells.
        ser_spktimes = []
        for sweep_no in range(I.shape[0]):
            ser_spktimes_singlesweep = []
            for ser_no in range(self.no_ser_neurons):
                if verbose:
                    print(
                        'Simulating 5HT neurons sweep {} of {} '
                        '{:.1f}%'.format(
                            sweep_no, I.shape[0],
                            100 * (ser_no + 1) / self.no_ser_neurons
                        )
                    )

                t, V, eta, v_T, spks = self.ser_mod[ser_no].simulate(
                    I[sweep_no, ser_no, :], self.ser_mod[ser_no].El
                )

                # Save spktimes/spktrains.
                ser_spktimes_singlesweep.append(spks)
                out.ser_spktrains[sweep_no, ser_no, :] = timeToIntVec(
                    spks, out.get_T(), self.dt
                )

                # Save sample traces.
                if ser_no < out.get_no_ser_examples():
                    tmp_results = {
                        't': t, 'V': V, 'eta': eta,
                        'v_T': v_T, 'I': ser_input[sweep_no, ser_no, :]
                    }

                    # Only save pre-initialized channels.
                    for key in out.ser_examples.keys():
                        if key == 'feedforward_input':
                            continue
                        else:
                            out.ser_examples[key][sweep_no, ser_no, :] = tmp_results[key]

            ser_spktimes.append(ser_spktimes_singlesweep)

        return ser_spktimes

    def simulate(self, out, ser_input = None, gaba_input = None, do_feedforward = True, verbose = True):
        """
        Perform GIF network simulation.

        Inputs:
            out (GIFnet_Simulation)
                -- GIFnet_Simulation in which to place simulation output.
            ser_input (3D array or None)
                -- External input for driving 5HT neurons in the network.
                Should have shape (no_sweeps, no_ser_neurons, timesteps).
                Set to None to skip simulating 5HT cells.
            gaba_input (3D array or None)
                -- External input for driving GABA neurons in the network.
                Should have shape (no_sweeps, no_gaba_neurons, timesteps).
                Set to None to skip simulating GABA cells.
            do_feedforward (bool, default True)
                -- Connect GABA and 5HT layers. If false, responses of both
                populations to external input are simulated, but GABA output
                is not given to 5HT neurons.

        Returns:
            Nothing. Results are saved directly in 'out'.
        """

        ### Input checks.
        if type(out) is not GIFnet_Simulation:
            raise TypeError(
                '\'out\' must be a GIFnet_Simulation object.'
            )
        if ser_input is None and gaba_input is None:
            raise ValueError(
                'One of ser_input or gaba_input must not be None.'
            )
        # Check number of timesteps.
        elif (ser_input is not None and gaba_input is not None
              and ser_input.shape[2] != gaba_input.shape[2]):
            raise ValueError(
                'ser_input no_timesteps ({}) must match gaba_input '
                'no_timesteps ({})'.format(
                    ser_input.shape[2], gaba_input.shape[2]
                )
            )
        # Check number of sweeps.
        elif (ser_input is not None and gaba_input is not None
              and ser_input.shape[0] != gaba_input.shape[0]):
            raise ValueError(
                'ser_input no_sweeps ({}) must match gaba_input no_sweeps ({})'.format(
                    ser_input.shape[0], gaba_input.shape[0]
                )
            )
        # Check number of neurons.
        elif ser_input is not None and ser_input.shape[1] != self.no_ser_neurons:
            raise ValueError(
                'ser_input second dim ({}) must match GIFnet.no_ser_neurons ({})'.format(
                    ser_input.shape[1], self.no_ser_neurons
                )
            )
        elif gaba_input is not None and gaba_input.shape[1] != self.no_gaba_neurons:
            raise ValueError(
                'gaba_input second dim ({}) must match GIFnet.no_gaba_neurons ({})'.format(
                    gaba_input.shape[1], self.no_gaba_neurons
                )
            )

        ### Set metaparams in 'out'.
        out.set_dt(self.dt)
        if ser_input is not None:
            out.set_T(int(ser_input.shape[2] * self.dt))
            out.set_no_sweeps(ser_input.shape[0])
        elif gaba_input is not None:
            out.set_T(int(gaba_input.shape[2] * self.dt))
            out.set_no_sweeps(gaba_input.shape[0])
        else:
            raise RuntimeError(
                'No inputs provided from which to get simulation length '
                'or number of sweeps.'
            )

        ### Run simulations.

        # Simulate GABA neurons (if applicable).
        if gaba_input is not None:
            out.set_no_gaba_neurons(self.no_gaba_neurons)
            out.init_gaba_spktrains()
            gaba_spktimes = self._simulate_gaba(out, gaba_input, verbose)

        # Simulate 5HT neurons (if applicable).
        if ser_input is not None:

            # Generate GABA input to 5HT cells based on GABA spks.
            if gaba_input is not None and do_feedforward:
                out.set_propagation_delay(self.propagation_delay)
                out.set_connectivity_matrix(self.connectivity_matrix)
                feedforward_input = self._convolve_feedforward(out, gaba_spktimes)
            else:
                feedforward_input = None

            # Simulate 5HT neurons.
            out.set_no_ser_neurons(self.no_ser_neurons)
            out.init_ser_spktrains()
            self._simulate_ser(out, ser_input, feedforward_input, verbose)


# Simple test for GIFnet
if __name__ == '__main__':

    import os

    from .GIF import GIF

    no_sweeps = 2
    no_ser_neurons = 5
    no_gaba_neurons = 10
    T = 100
    dt = 0.1

    connectivity_matrix = np.random.uniform(size = (no_ser_neurons, no_gaba_neurons))

    ser_input = 0.1 * np.ones((no_sweeps, no_ser_neurons, int(T / dt)), dtype = np.float32)
    gaba_input = 0.1 * np.ones((no_sweeps, no_gaba_neurons, int(T / dt)), dtype = np.float32)

    # Try initializing GIFnet.
    test_gifnet = GIFnet(
        ser_mod = [GIF(dt)] * no_ser_neurons,
        gaba_mod = [GIF(dt)] * no_gaba_neurons,
        gaba_kernel = [1, 1, 1, 0.5],
        propagation_delay = 1.,
        connectivity_matrix = connectivity_matrix,
        dt = dt
    )

    testfname = 'testgifnetsimfile.hdf5.test'
    meta_args = {
        'name': 'test sim',
        'T': T, 'dt': dt,
        'no_sweeps': no_sweeps,
        'no_ser_examples': 2,
        'no_gaba_examples': 3
    }
    with GIFnet_Simulation(testfname, **meta_args) as outfile:
        # Set channels to save in examples.
        outfile.init_gaba_examples()
        outfile.init_ser_examples(v_T = None)

        # Try running a simple simulation.
        test_gifnetsim = test_gifnet.simulate(
            outfile,
            ser_input = ser_input, gaba_input = gaba_input
        )

    # Try saving output.
    os.remove(testfname)
