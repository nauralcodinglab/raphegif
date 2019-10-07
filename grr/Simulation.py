#%% IMPORT MODULES

from __future__ import division

import warnings

import numpy as np
import h5py

from .Tools import timeToIntVec


#%% DEFINE CLASSES FOR REPRESENTING SIMULATIONS

class GIFnet_Simulation(h5py.File):

    """Represents a GIFnet simulation.

    Subclass of h5py.File.

    Suggested usage:

    sim = GIFnet_Simulation(
        'example.hdf5', 'Example simulation',
        T = 100., dt = 0.1, no_sweeps = 10,
        no_ser_neurons = 10,
        no_ser_examples = 3,
        no_gaba_neurons = 5,
        no_gaba_examples = 3,
        propagation_delay = 2.
    )

    sim.set_connectivity_matrix(connectivity_matrix)

    sim.init_ser_examples(**ser_examples)

    sim.init_ser_spktrains()
    for sweep_no in ser_spktrains.shape[0]:
        for cell_no in ser_spktrains.shape[1]:
            sim.ser_spktrains[sweep_no, cell_no, :] = ser_spktrains[sweep_no, cell_no, :]

    sim.init_gaba_examples(
        I = gaba_examples['I'],
        V = gaba_examples['V'],
        some_channel = gaba_examples['some_channel']
    )

    sim.init_gaba_spktrains(
        spktrains = gaba_spktrains
    )
    """

    def __init__(self, fname, name = None,
                 T = None, dt = None, no_sweeps = None,
                 no_ser_neurons = None,
                 no_ser_examples = None,
                 no_gaba_neurons = None,
                 no_gaba_examples = None,
                 propagation_delay = None,
                 **kwargs):

        """Create a new GIFnet_Simulation object.

        Inputs:
            fname (str)
                -- Name of file on disk in which to store
                contents of GIFnet_Simulation. (Equivalent
                to h5py.File's 'name' argument.)
            name (str)
                -- Meta-attribute with short description
                of experiment.
            T (float)
                -- Duration of each sweep (ms).
            dt (float)
                -- Timestep (ms).
            no_sweeps (int)
                -- Number of sweeps in simulation.
            no_ser_neurons, no_gaba_neurons (int)
                -- Total number of ser/gaba neurons in population.
                Spiketrains of this number of neurons are stored.
            no_ser_examples, no_gaba_examples (int)
                -- Number of neurons in population for which
                full traces are stored.
            propagation_delay (float)
                -- Delay between GABA spike and start of IPSC
                in 5HT neurons (ms).
            kwargs
                -- Keyword arguments to be passed to h5py.File
                initializer.
        """

        if kwargs.get('mode', 'a') not in ['r', 'a']:
            raise ValueError('\'mode\' must be \'r\' or \'a\'')

        super(GIFnet_Simulation, self).__init__(
            name = fname, mode = kwargs.pop('mode', 'a'), **kwargs
        )

        if name is not None:
            self.set_name(name)
        if T is not None:
            self.set_T(T)
        if dt is not None:
            self.set_dt(dt)
        if no_sweeps is not None:
            self.set_no_sweeps(no_sweeps)
        if no_ser_neurons is not None:
            self.set_no_ser_neurons(no_ser_neurons)
        if no_ser_examples is not None:
            self.set_no_ser_examples(no_ser_examples)
        if no_gaba_neurons is not None:
            self.set_no_gaba_neurons(no_gaba_neurons)
        if no_gaba_examples is not None:
            self.set_no_gaba_examples(no_gaba_examples)
        if propagation_delay is not None:
            self.set_propagation_delay(propagation_delay)

    ### Getters and setters for meta-attributes.
    def get_name(self):
        # 'name' is a short description, not a filename.
        if 'name' not in self.attrs.keys():
            raise KeyError('\'name\' not set.')
        else:
            return self.attrs['name']

    def set_name(self, val):
        # 'name' is a short description, not a filename.
        self.attrs['name'] = val

    def get_no_sweeps(self):
        if 'no_sweeps' not in self.attrs.keys():
            raise KeyError('\'no_sweeps\' not set.')
        else:
            return self.attrs['no_sweeps']

    def set_no_sweeps(self, val):
        self.attrs['no_sweeps'] = val

    def get_T(self):
        if 'T' not in self.attrs.keys():
            raise KeyError('\'T\' not set.')
        else:
            return self.attrs['T']

    def set_T(self, val):
        self.attrs['T'] = val

    def get_dt(self):
        if 'dt' not in self.attrs.keys():
            raise KeyError('\'dt\' not set.')
        else:
            return self.attrs['dt']

    def set_dt(self, val):
        self.attrs['dt'] = val

    def get_no_timesteps(self):
        if not ('dt' in self.attrs.keys()
                and 'T' in self.attrs.keys()):
            raise KeyError('\'dt\' and \'T\' must both be set.')
        else:
            return int(self.get_T() / self.get_dt() + 0.5)

    def get_no_ser_neurons(self):
        if 'no_ser_neurons' not in self.attrs.keys():
            raise KeyError('\'no_ser_neurons\' not set.')
        else:
            return self.attrs['no_ser_neurons']

    def set_no_ser_neurons(self, val):
        self.attrs['no_ser_neurons'] = val

    def get_no_ser_examples(self):
        if 'no_ser_examples' not in self.attrs.keys():
            raise KeyError('\'no_ser_examples\' not set.')
        else:
            return self.attrs['no_ser_examples']

    def set_no_ser_examples(self, val):
        self.attrs['no_ser_examples'] = val

    def get_no_gaba_neurons(self):
        if 'no_gaba_neurons' not in self.attrs.keys():
            raise KeyError('\'no_gaba_neurons\' not set.')
        else:
            return self.attrs['no_gaba_neurons']

    def set_no_gaba_neurons(self, val):
        self.attrs['no_gaba_neurons'] = val

    def get_no_gaba_examples(self):
        if 'no_gaba_examples' not in self.attrs.keys():
            raise KeyError('\'no_gaba_examples\' not set.')
        else:
            return self.attrs['no_gaba_examples']

    def set_no_gaba_examples(self, val):
        self.attrs['no_gaba_examples'] = val

    def get_propagation_delay(self):
        if 'propagation_delay' not in self.attrs.keys():
            raise KeyError('\'propagation_delay\' not set.')
        else:
            return self.attrs['propagation_delay']

    def set_propagation_delay(self, val):
        self.attrs['propagation_delay'] = val

    ### Getter and setter for connectivity matrix.
    def get_connectivity_matrix(self):
        if 'connectivity_matrix' not in self.keys():
            raise AttributeError(
                'connectivity_matrix not set.'
            )
        else:
            return self['connectivity_matrix']

    def set_connectivity_matrix(self, arr, infer_pop_size = True):
        """Create connectivity matrix for feedforward connections

        Inputs:
            arr (2D array)
                -- 2D array with dimensionality
                [no_ser_neurons, no_gaba_neurons] specifyinging
                gaba->ser connections.
            infer_pop_size (bool, default True)
                -- Optionally, set no_ser_neurons and no_gaba_neurons
                based on dimensionality of arr. Throws a warning
                if these are already set to different values.
        """

        ### Input checks.
        # Check input dimensionality.
        if arr.ndim != 2:
            raise ValueError(
                'Array must be 2D, not {}D.'.format(arr.ndim)
            )

        if (hasattr(self.attrs, 'no_ser_neurons') and
            self.attrs['no_ser_neurons'] != arr.shape[0]):
            warnings.warn(
                'Apparent no_ser_neurons in arr ({}) differs '
                'from instance no_ser_neurons ({}).'
                ''.format(arr.shape[0], self.get_no_ser_neurons())
            )
            ser_shape_flag = True
        else:
            ser_shape_flag = False

        if (hasattr(self.attrs, 'no_gaba_neurons') and
            self.attrs['no_gaba_neurons'] != arr.shape[0]):
            warnings.warn(
                'Apparent no_gaba_neurons in arr ({}) differs '
                'from instance no_gaba_neurons ({}).'
                ''.format(arr.shape[0], self.get_no_gaba_neurons())
            )
            gaba_shape_flag = True
        else:
            gaba_shape_flag = False

        # Halt if not allowed to overwrite pop sizes and arr dims
        # do not match pre-set pop sizes.
        if not infer_pop_size and (ser_shape_flag or gaba_shape_flag):
            raise ValueError(
                'Instance no_ser_neurons and/or no_gaba_neurons do '
                'not match arr shape {}.'.format(arr.shape)
            )

        # Initialize connectivity_matrix.
        self.create_dataset(
            'connectivity_matrix', data = arr,
            dtype = np.float32, compression = 5
        )

        # Ensure that pop sizes are up to date.
        self.set_no_ser_neurons(arr.shape[0])
        self.set_no_gaba_neurons(arr.shape[1])

    ### Properties and initializers for recorded signals.
    @property
    def ser_spktrains(self):
        if 'ser' not in self.keys() or 'spktrains' not in self['ser'].keys():
            raise AttributeError(
                'ser_spktrains must be initialized via init_ser_spktrains '
                'first.'
            )
        else:
            return self['ser/spktrains']

    def init_ser_spktrains(self, spktrains = None, spktimes = None):
        """Initialize ser spiketrains as an indicator tensor

        Save spiketrains as an indicator tensor, starting
        from a tensor of spiketrains or list of lists.
        Note that both types of input are equivalent, but
        at most one should be passed at a time.

        If neither spktrains nor spktimes is passed in, an empty
        spktrain array is simply created with the correct shape.

        ser_pktrains can be written and read via instance
        ser_spktrains attribute.

        Inputs:
            spktrains (3D array, or None)
                -- 3D indicator tensor (1 when a spike
                occurs, 0 otherwise) with dimensionality
                [sweeps, cells, timesteps].
            spktimes (nested list of depth == 3, or None)
                -- Nested list laid out according to
                [sweep][cell][spike_number] with times of
                each spike for each cell on each sweep.
        """

        if spktimes is not None and spktrains is not None:
            raise ValueError(
                'Only spktimes or spktrains should be provided, '
                'not both.'
            )

        sergroup = self.require_group('ser')
        sspks = sergroup.create_dataset(
            'spktrains',
            shape = (self.get_no_sweeps(),
                     self.get_no_ser_neurons(),
                     self.get_no_timesteps()),
            dtype = np.int8, compression = 5
        )

        # Case that spktrains have been provided directly.
        if spktrains is not None:
            sspks[:, :, :] = spktrains

        # Case that nested list of spktimes has been provided.
        elif spktimes is not None:
            for i in range(len(spktimes)):
                for j in range(len(spktimes[0])):
                    sspks[i, j, :] = timeToIntVec(
                        spktimes[i][j], self.get_T(), self.get_dt()
                    )

    @property
    def ser_examples(self):
        if 'ser' not in self.keys() or 'examples' not in self['ser'].keys():
            raise AttributeError(
                'ser_examples must be initialized via init_ser_examples '
                'first.'
            )
        else:
            return self['ser/examples']

    def init_ser_examples(self, I = None, V = None,
                          feedforward_input = None,
                          **kwargs):
        """Initialize ser example traces

        Any inputs set to None will be initialized as empty
        arrays.

        Inputs:
            I (3D array or None)
                -- 3D array with dimensionality
                [sweeps, cells, timesteps] to initialize
                input current channel.
            V (3D array or None)
                -- 3D array with dimensionality
                [sweeps, cells, timesteps] to initialize
                recorded voltage channel.
            feedforward_input (3D array or None)
                -- 3D array with dimensionality
                [sweeps, cells, timesteps] to initialize
                 recorded gaba->ser feedforward_input.
            kwargs (3D array or None)
                -- Identically-shaped 3D arrays for
                any other channels to initialize.
        """
        sergroup = self.require_group('ser')
        serex = sergroup.require_group('examples')

        pairs = kwargs.copy()
        pairs.update({
            'I': I, 'V': V,
            'feedforward_input': feedforward_input
        })

        for key, val in pairs.iteritems():

            # Initialize with data, if available.
            if val is not None:
                serex.create_dataset(
                    key, data = val,
                    shape = (self.get_no_sweeps(),
                             self.get_no_ser_examples(),
                             self.get_no_timesteps()),
                    dtype = np.float32, compression = 5
                )

            # Initialize empty if no data available.
            else:
                serex.create_dataset(
                    key, fillvalue = 0,
                    shape = (self.get_no_sweeps(),
                             self.get_no_ser_examples(),
                             self.get_no_timesteps()),
                    dtype = np.float32, compression = 5
                )

    @property
    def gaba_examples(self):
        if 'gaba' not in self.keys() or 'examples' not in self['gaba'].keys():
            raise AttributeError(
                'gaba_examples must be initialized via init_gaba_examples '
                'first.'
            )
        else:
            return self['gaba/examples']

    def init_gaba_examples(self, I = None, V = None, **kwargs):
        """Initialize gaba example traces

        Any inputs set to None will be initialized as empty
        arrays.

        Inputs:
            I (3D array or None)
                -- 3D array with dimensionality
                [sweeps, cells, timesteps] to initialize
                input current channel.
            V (3D array or None)
                -- 3D array with dimensionality
                [sweeps, cells, timesteps] to initialize
                recorded voltage channel.
            kwargs (3D array or None)
                -- Identically-shaped 3D arrays for
                any other channels to initialize.
        """
        gabagroup = self.require_group('gaba')
        gabaex = gabagroup.require_group('examples')

        pairs = kwargs.copy()
        pairs.update({'I': I, 'V': V})

        for key, val in pairs.iteritems():

            # Initialize with data, if available.
            if val is not None:
                gabaex.create_dataset(
                    key, data = val,
                    shape = (self.get_no_sweeps(),
                             self.get_no_gaba_examples(),
                             self.get_no_timesteps()),
                    dtype = np.float32, compression = 5
                )

            # Initialize empty if no data available.
            else:
                gabaex.create_dataset(
                    key, fillvalue = 0,
                    shape = (self.get_no_sweeps(),
                             self.get_no_gaba_examples(),
                             self.get_no_timesteps()),
                    dtype = np.float32, compression = 5
                )

    @property
    def gaba_spktrains(self):
        if 'gaba' not in self.keys() or 'spktrains' not in self['gaba'].keys():
            raise AttributeError(
                'gaba_spktrains must be initialized via init_gaba_spktrains '
                'first.'
            )
        else:
            return self['gaba/spktrains']

    def init_gaba_spktrains(self, spktrains = None, spktimes = None):
        """Initialize gaba spiketrains as an indicator tensor

        Save spiketrains as an indicator tensor, starting
        from a tensor of spiketrains or list of lists.
        Note that both types of input are equivalent, but
        only one should be passed at a time.

        If neither spktrains nor spktimes is passed in, an empty
        spktrain array is simply created with the correct shape.

        gaba_pktrains can be written and read via instance
        gaba_spktrains attribute.

        Inputs:
            spktrains (3D array, or None)
                -- 3D indicator tensor (1 when a spike
                occurs, 0 otherwise) with dimensionality
                [sweeps, cells, timesteps].
            spktimes (nested list of depth == 3, or None)
                -- Nested list laid out according to
                [sweep][cell][spike_number] with times of
                each spike for each cell on each sweep.
        """

        if spktimes is not None and spktrains is not None:
            raise ValueError(
                'Only spktimes or spktrains should be provided, '
                'not both.'
            )

        gabagroup = self.require_group('gaba')
        gspks = gabagroup.create_dataset(
            'spktrains',
            shape = (self.get_no_sweeps(),
                     self.get_no_gaba_neurons(),
                     self.get_no_timesteps()),
            dtype = np.int8, compression = 5
        )

        # Case that spktrains have been provided directly.
        if spktrains is not None:
            gspks[:, :, :] = spktrains

        # Case that nested list of spktimes has been provided.
        elif spktimes is not None:
            for i in range(len(spktimes)):
                for j in range(len(spktimes[0])):
                    gspks[i, j, :] = timeToIntVec(
                        spktimes[i][j], self.get_T(), self.get_dt()
                    )

    ### Data processing and support arrays.
    def get_ser_spktimes(self):
        """Get nested list of 5HT neuron spktimes.

        Nested list should be indexed according
        to [sweep_no][cell_no][spk_no].
        """

        spktimes = []
        for sweep_no in range(self.get_no_sweeps()):
            spktimes_singlesweep = []
            for cell_no in range(self.get_no_ser_neurons()):
                spktimes_singlesweep.append(
                    np.where(
                        self.ser_spktrains[sweep_no, cell_no, :] > 0.5
                    )[0] * self.get_dt()
                )
            spktimes.append(spktimes_singlesweep)
        return spktimes

    def get_gaba_spktimes(self):
        """Get nested list of GABA neuron spktimes.

        Nested list should be indexed according
        to [sweep_no][cell_no][spk_no].
        """

        spktimes = []
        for sweep_no in range(self.get_no_sweeps()):
            spktimes_singlesweep = []
            for cell_no in range(self.get_no_gaba_neurons()):
                spktimes_singlesweep.append(
                    np.where(
                        self.gaba_spktrains[sweep_no, cell_no, :] > 0.5
                    )[0] * self.get_dt()
                )
            spktimes.append(spktimes_singlesweep)
        return spktimes

    def get_t_vec(self):
        """Return a time support vector (ms).
        """
        t_vec = np.arange(0, self.get_T(), self.get_dt())

        # Shape checks.
        if 'ser' in self.keys() and 'spktrains' in self['ser'].keys():
            assert self.ser_spktrains.shape[2] == len(t_vec), 'Bad t_vec length ({})'.format(len(t_vec))
        if 'gaba' in self.keys() and 'spktrains' in self['gaba'].keys():
            assert self.gaba_spktrains.shape[2] == len(t_vec), 'Bad t_vec length ({})'.format(len(t_vec))

        return t_vec

    def get_ser_examples_supp(self):
        """Get support array for ser_examples.
        """
        return np.broadcast_to(
            self.get_t_vec(),
            self.ser_examples[self.ser_examples.keys()[0]].shape
        )

    def get_gaba_examples_supp(self):
        """Get support array for gaba_examples.
        """
        return np.broadcast_to(
            self.get_t_vec(),
            self.gaba_examples[self.gaba_examples.keys()[0]].shape
        )

