#%% IMPORT MODULES

from __future__ import division
from copy import deepcopy

import numpy as np

from .Tools import timeToIntVec
from .Tools import validate_array_ndim, validate_matching_axis_lengths
from .Simulation import GIFnet_Simulation
from .ModelStimulus import ModelStimulus


#%% DEFINE GIF NETWORK CLASS

class GIFnet(object):

    def __init__(self, name=None, dt=0.1,
                 ser_mod=None, gaba_mod=None,
                 propagation_delay=0., connectivity_matrix=None,
                 gaba_kernel=None, gaba_reversal=None):
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
            gaba_reversal (float or None)
                -- Reversal potential (mV) of GABA conductance. Set to `None`
                to treat IPSC kernel as a current.
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
        self.__tmp_simulation_timesteps = None
        self.__tmp_simulation_sweeps = None

        self.ser_mod = ser_mod
        self.gaba_mod = gaba_mod
        self.propagation_delay = propagation_delay
        self.connectivity_matrix = connectivity_matrix
        self.gaba_kernel = gaba_kernel
        self.gaba_reversal = gaba_reversal

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

    @property
    def __tmp_simulation_duration(self):
        if self.__tmp_simulation_timesteps is not None:
            return self.__tmp_simulation_timesteps * self.dt
        else:
            raise AttributeError(
                'Duration undefined when `__tmp_simulation_timesteps` is not '
                'set.'
            )

    def simulate(self, out, ser_input=None, gaba_input=None, do_feedforward=True, verbose=True):
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
        if (
            (do_feedforward and gaba_input is not None)
            and self.connectivity_matrix is None
        ):
            raise ValueError(
                'Attribute `connectivity_matrix` cannot be `None` for '
                'simulations with feed-forward input.'
            )
        self._initialize_simulation_container(
            out, ser_input, gaba_input, do_feedforward
        )

        if gaba_input is not None:
            gaba_spktimes = self._simulate_gaba(out, gaba_input, verbose)
            if do_feedforward:
                feedforward_input = self._convolve_feedforward(
                    out, gaba_spktimes
                )

        if ser_input is not None:
            self._simulate_ser(out, ser_input, feedforward_input, verbose)

    def _simulate_gaba(self, out, gaba_input, verbose=False):
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
        # Check input shape.
        self._get_valid_input_dims(ser_input=None, gaba_input=gaba_input)

        gaba_spktimes = []
        for sweep_no in range(self.__tmp_simulation_sweeps):
            gaba_spktimes_singlesweep = []
            for gaba_no in range(self.no_gaba_neurons):
                if verbose:
                    print(
                        'Simulating gaba neurons sweep {} of {} '
                        '{:.1f}%'.format(
                            sweep_no, self.__tmp_simulation_sweeps,
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
                    spks, self.__tmp_simulation_duration, self.dt
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
        if self.connectivity_matrix is None:
            raise AttributeError(
                'Feed-forward inputs cannot be computed if GIFnet '
                'connectivity_matrix is not set.'
            )

        #TODO: extract convolution method and separate from prop delay and projection steps.
        # Convert gaba_spktimes to convolved spktrains.
        gaba_conv_spks = np.empty(
            (len(gaba_spktimes),
             self.no_gaba_neurons,
             self.__tmp_simulation_timesteps),
            dtype=np.float32
        )
        for sweep_no in range(len(gaba_spktimes)):
            for gaba_no in range(self.no_gaba_neurons):
                gaba_conv_spks[sweep_no, gaba_no, :] = np.convolve(
                    timeToIntVec(
                        gaba_spktimes[sweep_no][gaba_no], self.__tmp_simulation_duration, self.dt
                    ),
                    self.gaba_kernel, 'same'
                ).astype(np.float32)

        gaba_conv_spks = self._apply_propagation_delay(gaba_conv_spks)
        feedforward_input = self._project_onto_target_population(gaba_conv_spks)

        # Save synaptic input trains in 'out'.
        out.ser_examples['feedforward_input'][...] = (
            feedforward_input[:, :out.get_no_ser_examples(), :]
        )

        return feedforward_input

    def _apply_propagation_delay(self, feedforward_array):
        """Apply instance propagation delay to array of feed-forward input."""
        if self.propagation_delay is None:
            raise ValueError(
                'Cannot apply `propagation_delay=None`, set '
                '`propagation_delay=0.` explicitly if that is what you want.'
            )

        feedforward_array = deepcopy(feedforward_array)
        feedforward_array = np.roll(
            feedforward_array, int(self.propagation_delay / self.dt),
            axis=2
        )
        feedforward_array[:, :, :int(self.propagation_delay / self.dt)] = 0

        return feedforward_array

    def _project_onto_target_population(self, feedforward_array):
        projection = np.moveaxis(
            np.tensordot(
                self.connectivity_matrix,
                np.moveaxis(feedforward_array, 0, -1),
                axes=1
            ),
            -1, 0
        )
        return projection

    def _simulate_ser(self, out, ser_input, feedforward_input=None, verbose=False):
        """Simulate response of 5HT neurons to ser_input.

        Input:
            out (GIFnet_Simulation)
                -- Object in which to store simulation results directly.
            ser_input (3D array)
                -- External input to 5HT cells. Dimensionality [sweeps, neurons, time].
            feedforward_input (2D array)
                -- Feedforward input to 5HT cells from GABA population.
                Must have same shape as ser_input.
            verbose (bool, default False)
                -- Print information about progress.

        Returns:
            Dict with spktimes (list of lists of spktimes from each cell)
            and example (sample trace).
        """
        # Check input shapes.
        self._get_valid_input_dims(ser_input=ser_input, gaba_input=None)
        self._get_valid_input_dims(ser_input=feedforward_input, gaba_input=None)

        # Loop over sweeps and cells.
        ser_spktimes = []
        for sweep_no in range(self.__tmp_simulation_sweeps):
            ser_spktimes_singlesweep = []
            for ser_no in range(self.no_ser_neurons):
                if verbose:
                    print(
                        'Simulating 5HT neurons sweep {} of {} '
                        '{:.1f}%'.format(
                            sweep_no, self.__tmp_simulation_sweeps,
                            100 * (ser_no + 1) / self.no_ser_neurons
                        )
                    )

                mod_stim = self._assemble_model_stimulus(
                    ser_input[sweep_no, ser_no, :],
                    feedforward_input[sweep_no, ser_no, :]
                )
                t, V, eta, v_T, spks = self.ser_mod[ser_no].simulate(
                    mod_stim, self.ser_mod[ser_no].El
                )

                # Save spktimes/spktrains.
                ser_spktimes_singlesweep.append(spks)
                out.ser_spktrains[sweep_no, ser_no, :] = timeToIntVec(
                    spks, self.__tmp_simulation_duration, self.dt
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

    def _assemble_model_stimulus(self, direct_input, feedforward_input):
        mod_stim = ModelStimulus(self.dt)

        # Direct input is always current-based.
        mod_stim.appendCurrents(direct_input)

        # Feed-forward input may be conductance-based, depending on whether
        # `gaba_reversal` is set.
        if self.gaba_reversal is None:
            mod_stim.appendCurrents(feedforward_input)
        else:
            mod_stim.appendConductances(feedforward_input, [self.gaba_reversal])

        return mod_stim

    def _initialize_simulation_container(self, out, ser_input, gaba_input, do_feedforward):
        valid_dims = self._get_valid_input_dims(ser_input, gaba_input)
        self.__tmp_simulation_timesteps = valid_dims['timesteps']
        self.__tmp_simulation_sweeps = valid_dims['sweeps']

        out.set_dt(self.dt)
        out.set_T(self.dt)
        out.set_no_sweeps(self.__tmp_simulation_sweeps)

        if gaba_input is not None:
            out.set_no_gaba_neurons(self.no_gaba_neurons)
            out.init_gaba_spktrains()
            if do_feedforward:
                out.set_propagation_delay(self.propagation_delay)
                out.set_connectivity_matrix(self.connectivity_matrix)

        if ser_input is not None:
            out.set_no_ser_neurons(self.no_ser_neurons)
            out.init_ser_spktrains()

    def _get_valid_input_dims(self, ser_input, gaba_input):
        """Get dimensionality of inputs to `simulate()` or raise an exception."""
        for label, arr in zip(['ser_input', 'gaba_input'], [ser_input, gaba_input]):
            validate_array_ndim(label, arr, 3)

        # Extract input dimensions from whichever array is provided.
        if ser_input is not None and gaba_input is not None:
            validate_matching_axis_lengths([ser_input, gaba_input], [0, 2])  # Should match along axis 0 (sweeps) and 2 (time).
            valid_dims = {
                'sweeps': np.shape(ser_input)[0],
                'timesteps': np.shape(ser_input)[2],
                'no_ser_neurons': np.shape(ser_input)[1],
                'no_gaba_neurons': np.shape(gaba_input)[1]
            }
        elif ser_input is not None and gaba_input is None:
            valid_dims = {
                'sweeps': np.shape(ser_input)[0],
                'timesteps': np.shape(ser_input)[2],
                'no_ser_neurons': np.shape(ser_input)[1],
                'no_gaba_neurons': 0
            }
        elif ser_input is None and gaba_input is not None:
            valid_dims = {
                'sweeps': np.shape(gaba_input)[0],
                'timesteps': np.shape(gaba_input)[2],
                'no_ser_neurons': 0,
                'no_gaba_neurons': np.shape(gaba_input)[1]
            }
        else:
            RuntimeError('Cannot get here.')

        # Ensure input dimensions match number of models in instance
        # if input has been provided.
        for label in ['no_ser_neurons', 'no_gaba_neurons']:
            if (
                (valid_dims[label] != 0)
                and (valid_dims[label] != getattr(self, label))
            ):
                raise ValueError(
                    'Expected input axis 1 `{}` length {} to match instance '
                    'number of models {}'.format(
                        label, valid_dims[label], getattr(self, label)
                    )
                )

        return valid_dims

    def clear_interpolated_filters(self):
        """Clear cached interpolated spike-triggered filters.

        Eta and gamma spike-triggered filters are interpolated to potentially
        large arrays during simulations. The filters are re-interpolated as
        needed, so saving the interpolated versions is unnecessary. Clearing
        the interpolated filters can reduce object size significantly, and is
        recommended to do before saving the model to disk.

        """
        if self.ser_mod is not None:
            for i in range(len(self.ser_mod)):
                self.ser_mod[i].eta.clearInterpolatedFilter()
                self.ser_mod[i].gamma.clearInterpolatedFilter()

        if self.gaba_mod is not None:
            for i in range(len(self.gaba_mod)):
                self.gaba_mod[i].eta.clearInterpolatedFilter()
                self.gaba_mod[i].gamma.clearInterpolatedFilter()





# Simple test for GIFnet
if __name__ == '__main__':

    import os

    from .GIF import GIF

    no_sweeps = 2
    no_ser_neurons = 5
    no_gaba_neurons = 10
    T = 100
    dt = 0.1

    connectivity_matrix = np.random.uniform(size=(no_ser_neurons, no_gaba_neurons))

    ser_input = 0.1 * np.ones((no_sweeps, no_ser_neurons, int(T / dt)), dtype=np.float32)
    gaba_input = 0.1 * np.ones((no_sweeps, no_gaba_neurons, int(T / dt)), dtype=np.float32)

    # Try initializing GIFnet.
    test_gifnet = GIFnet(
        ser_mod=[GIF(dt)] * no_ser_neurons,
        gaba_mod=[GIF(dt)] * no_gaba_neurons,
        gaba_kernel=[1, 1, 1, 0.5],
        propagation_delay=1.,
        connectivity_matrix=connectivity_matrix,
        dt=dt
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
        outfile.init_ser_examples(v_T=None)

        # Try running a simple simulation.
        test_gifnetsim = test_gifnet.simulate(
            outfile,
            ser_input=ser_input, gaba_input=gaba_input
        )

    # Try saving output.
    os.remove(testfname)
