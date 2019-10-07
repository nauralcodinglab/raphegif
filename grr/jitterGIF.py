import math
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.linalg import inv
from numpy import nan, NaN
import numba as nb
from scipy.stats import binned_statistic
import weave

from .GIF import GIF
from .Filter_Rect import Filter_Rect_LogSpaced
from .Trace import Trace
from .Tools import reprint


class jitterGIF(GIF):

    """
    Generalized Integrate and Fire model defined in Pozzorini et al. PLOS Comp. Biol. 2015

    Spike are produced stochastically with firing intensity:

    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),

    where the membrane potential dynamics is given by:

    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)

    and the firing threshold V_T is given by:

    V_T = Vt_star + sum_j gamma(t-\hat t_j)

    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):

        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)

        # Define model parameters
        self.gl = 1.0/100.0        # nS, leak conductance
        self.C = 20.0*self.gl     # nF, capacitance
        self.El = -65.0            # mV, reversal potential

        # Define attributes to store goodness-of-fit
        self.var_explained_dV = 0
        self.var_explained_V = 0

        # Define attributes to store data used during fitting
        self.I_data = 0

        self.dV_data = 0
        self.dV_fitted = 0

        self.V_data = 0
        self.V_sim = 0
        self.m_sim = 0
        self.h_sim = 0
        self.n_sim = 0

        # Define attributes related to K_conductance gating parameters
        self.m_Vhalf = None
        self.m_k = None
        self.m_tau = None

        self.h_Vhalf = None
        self.h_k = None
        self.h_tau = None

        self.n_Vhalf = None
        self.n_k = None
        self.n_tau = None

        self.E_K = None

        self.gbar_K1 = 0
        self.gbar_K2 = 0

    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Spiking model
    ########################################################################################################

    def simulateSpikingResponse(self, I, dt):
        """
        Subthreshold model does not spike.
        """

        raise RuntimeError('Subthreshold model does not spike.')

    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Threshold Model
    ########################################################################################################

    def simulateVoltageResponse(self, I, dt):

        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return (spks_times, V, V_T)

    ########################################################################################################
    # SYNAPTIC INPUT RELATED METHODS
    ########################################################################################################

    def _convolveSynapticKernel(self, signal, ampli, tau_rise, tau_decay, kernel_length):

        # Convert to timesteps
        kernel_length = int(kernel_length / self.dt)
        tau_rise /= self.dt
        tau_decay /= self.dt

        # Generate synaptic waveform kernel
        kernel_support = np.arange(0, kernel_length)
        kernel = np.exp(-kernel_support/tau_decay) - np.exp(-kernel_support/tau_rise)
        kernel /= kernel.max()
        kernel *= ampli

        # Convolve with signal
        if signal.ndim == 1:
            signal.reshape((-1, 1))

        for i in range(signal.shape[1]):
            signal[:, i] = np.convolve(signal[:, i], kernel, 'same')

        return signal

    def initializeSynapses(self, no_synapses, ampli, tau_rise, tau_decay, duration, arrival_mean, arrival_SD, random_seed=None):

        synaptic_input = np.zeros((int(duration/self.dt), no_synapses), dtype=np.float64)
        synaptic_tvec = np.arange(0, int(synaptic_input.shape[0] * self.dt), self.dt)

        assert synaptic_input.shape[0] == len(synaptic_tvec)

        kernel_length = tau_decay*5
        compensated_arrival_mean = arrival_mean + kernel_length / 2.  # Needed because otherwise 'arrivals' occur in the middle of the synaptic waveform due to how convolution works.

        np.random.seed(random_seed)
        arrival_times = np.random.normal(compensated_arrival_mean, arrival_SD, no_synapses)
        arrival_inds = (arrival_times / self.dt).astype(np.int32)

        for syn_ind, arriv_ind in enumerate(arrival_inds):
            synaptic_input[arriv_ind, syn_ind] = 1.

        synaptic_input = self._convolveSynapticKernel(synaptic_input, ampli, tau_rise, tau_decay, kernel_length)

        # Assign output to jitterGIF attributes
        self.synaptic_input = synaptic_input

    def getSummatedSynapticInput(self):

        return self.synaptic_input.sum(axis=1)

    def getSynapticSupport(self, matrix_format=False):
        """
        Return a time support vector (or support matrix) for synaptic inputs.

        Note that the support matrix is just a time vector broadcasted to the shape of jitterGIF.synaptic_input.
        """

        support = np.arange(0, int(self.synaptic_input.shape[0] * self.dt), self.dt)
        assert len(support) == self.synaptic_input.shape[0]

        if matrix_format:
            support = np.broadcast_to(support.reshape((-1, 1)), self.synaptic_input.shape)

        return support

    def plotSynaptic(self):

        plt.figure()

        supp = self.getSynapticSupport(True)

        for i in range(self.synaptic_input.shape[1]):
            plt.subplot(self.synaptic_input.shape[1] + 1, 1, i + 1)
            plt.plot(supp[:, i], self.synaptic_input[:, i], 'k-', linewidth=0.5)

        plt.subplot(self.synaptic_input.shape[1] + 1, 1, self.synaptic_input.shape[1] + 1)
        plt.plot(supp[:, 0], self.getSummatedSynapticInput(), 'r-')

        plt.subplots_adjust(hspace=0)
        plt.show()

    def simulateSynaptic(self, V_rest, current_offset=None, spiking=False):
        """
        Essentially a switch to select between jitterGIF.simulate (for subthreshold simulations) and jitterGIF._simulateHardThreshold (for spiking simulations).
        """

        if current_offset is None:
            current_offset = 0

        if not spiking:

            return self.simulate(self.getSummatedSynapticInput() + current_offset, V_rest)

        else:

            return self._simulateHardThreshold(self.getSummatedSynapticInput() + current_offset, V_rest)

    def _simulateHardThreshold(self, I, V0):
        """
        Simulates spiking response using a hard threshold/hard reset model based on the subthreshold model.
        """

        # Input parameters
        p_T = len(I)
        p_dt = self.dt
        p_Vthresh = np.float64(self.Vthresh)
        p_Vreset = np.float64(self.Vreset)

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El

        p_m_Vhalf = self.m_Vhalf
        p_m_k = self.m_k
        p_m_tau = self.m_tau

        p_h_Vhalf = self.h_Vhalf
        p_h_k = self.h_k
        p_h_tau = self.h_tau

        p_n_Vhalf = self.n_Vhalf
        p_n_k = self.n_k
        p_n_tau = self.n_tau

        p_E_K = self.E_K

        p_gbar_K1 = self.gbar_K1
        p_gbar_K2 = self.gbar_K2

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")

        m = np.zeros_like(V, dtype="double")
        h = np.zeros_like(V, dtype="double")
        n = np.zeros_like(V, dtype="double")

        spks = np.zeros_like(V, dtype="double")

        # Set initial condition
        V[0] = V0

        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)

        code = """
                #include <math.h>
                #include <stdio.h>


                // DECLARE IMPORTED PARAMETERS

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);
                float Vthresh    = float(p_Vthresh);
                float Vreset     = float(p_Vreset);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);

                float m_Vhalf    = float(p_m_Vhalf);
                float m_k        = float(p_m_k);
                float m_tau      = float(p_m_tau);

                float h_Vhalf    = float(p_h_Vhalf);
                float h_k        = float(p_h_k);
                float h_tau      = float(p_h_tau);

                float n_Vhalf    = float(p_n_Vhalf);
                float n_k        = float(p_n_k);
                float n_tau      = float(p_n_tau);

                float E_K        = float(p_E_K);

                float gbar_K1    = float(p_gbar_K1);
                float gbar_K2    = float(p_gbar_K2);


                // DECLARE ADDITIONAL VARIABLES

                float m_inf_t;
                float h_inf_t;
                float n_inf_t;

                float DF_K_t;
                float gk_1_term;
                float gk_2_term;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = 1/(1 + exp(-m_k * (V[t-1] - m_Vhalf)));
                    m[t] = m[t-1] + dt/m_tau*(m_inf_t - m[t-1]);

                    // INTEGRATE h GATE
                    h_inf_t = 1/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = 1/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE K CONDUCTANCES
                    DF_K_t = V[t-1] - E_K;
                    gk_1_term = -DF_K_t * m[t] * h[t] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gk_1_term + gk_2_term);


                    // DO SPIKING STUFF IF NECESSARY
                    if (V[t] > Vthresh){

                    // Log spike
                    spks[t] = 1;
                    V[t] = 0;

                    // Reset condition
                    if (t+1 < T_ind){
                        V[t + 1] = Vreset;

                        // Leave nonlinearities unperturbed
                        m[t + 1] = m[t];
                        h[t + 1] = h[t];
                        n[t + 1] = n[t];
                    }

                    // Increment t
                    t = t + 1;

                    }

                }

                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_tau',
                'p_h_Vhalf', 'p_h_k', 'p_h_tau',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau',
                'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V', 'I', 'm', 'h', 'n',
                'p_Vthresh', 'p_Vreset', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        return (time, V, m, h, n, spks)

    def multiSim(self, jitters, gk2s, Els, no_reps=50, duration=800,
                 arrival_time=150, tau_rise=1, tau_decay=15,
                 ampli=0.010, no_syn=10, verbose=False):
        """
        sample_syn[t, syn, rep, jitter]
        Vsub[t, rep, jitter, gk2, El]
        n[t, rep, jitter, gk2, El]
        """

        sim_output = {
        'jitters': jitters,
        'gk2s': gk2s,
        'Els': Els,
        'sample_syn': np.empty((int(duration / self.dt), no_syn, no_reps, len(jitters)), dtype=np.float64),
        'Vsub': np.empty((int(duration / self.dt), no_reps, len(jitters), len(gk2s), len(Els)), dtype=np.float64),
        'n': np.empty((int(duration / self.dt), no_reps, len(jitters), len(gk2s), len(Els)), dtype=np.float64)
        #'pspk': np.zeros((int(duration / self.dt), len(jitters), len(gk2s)), dtype = np.float64),
        #'no_spks': np.empty((no_reps, len(jitters), len(gk2s)), dtype = np.int32)
        }

        simJGIF = deepcopy(self)

        for r_ in range(no_reps):

            if verbose:
                print '\rSimulating {}%'.format(100 * (r_ + 1)/no_reps),

            for j_ in range(len(jitters)):

                simJGIF.initializeSynapses(no_syn, ampli, tau_rise, tau_decay, duration, arrival_time, jitters[j_], r_)

                sim_output['sample_syn'][:, :, r_, j_] = simJGIF.synaptic_input

                for g_ in range(len(gk2s)):

                    simJGIF.gbar_K2 = gk2s[g_]

                    for e_ in range(len(Els)):

                        #simJGIF.El = Els[e_]

                        _, Vsub_tmp, m_tmp, h_tmp, n_tmp = simJGIF.simulateSynaptic(Els[e_], simJGIF.I_to_inject(Els[e_]), False)

                        #sim_output['no_spks'][r_, j_, g_] = spks_tmp.sum()
                        #sim_output['pspk'][:, j_, g_] += spks_tmp / no_reps

                        sim_output['Vsub'][:, r_, j_, g_, e_] = Vsub_tmp
                        sim_output['n'][:, r_, j_, g_, e_] = n_tmp

        return sim_output

    def I_to_inject(self, V):
        """
        Compute the amount of constant current to inject to obtain a steady-state voltage of V
        """

        I = V * (self.gl + self.gbar_K2 * self.nInf(V)) - (self.gl * self.El + self.gbar_K2 * self.nInf(V) * self.E_K)

        return float(I)

    ########################################################################################################
    # METHODS FOR K_CONDUCTANCE GATES
    ########################################################################################################

    def mInf(self, V):
        """Compute the equilibrium activation gate state of the potassium conductance.
        """

        return 1/(1 + np.exp(-self.m_k * (V - self.m_Vhalf)))

    def hInf(self, V):
        """Compute the equilibrium state of the inactivation gate of the potassium conductance.
        """

        return 1/(1 + np.exp(-self.h_k * (V - self.h_Vhalf)))

    def nInf(self, V):
        """Compute the equilibrium state of the non-inactivating conductance.
        """

        return 1/(1 + np.exp(-self.n_k * (V - self.n_Vhalf)))

    def computeGating(self, V, inf_vec, tau):
        """
        Compute the state of a gate over time.

        Wrapper for _computeGatingInternal, which is a nb.jit-accelerated static method.
        """

        return self._computeGatingInternal(V, inf_vec, tau, self.dt)

    @staticmethod
    @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64, nb.float64))
    def _computeGatingInternal(V, inf_vec, tau, dt):
        """
        Internal method called by computeGating.
        """

        output = np.empty_like(V, dtype=np.float64)
        output[0] = inf_vec[0]

        for i in range(1, len(V)):

            output[i] = output[i - 1] + (inf_vec[i - 1] - output[i - 1])/tau * dt

        return output

    def getDF_K(self, V):
        """
        Compute driving force on K based on SubthreshGIF_K.E_K and V.
        """

        return V - self.E_K

    ########################################################################################################
    # METHODS FOR NUMERICAL SIMULATIONS
    ########################################################################################################

    def simulate(self, I, V0):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        """

        # Input parameters
        p_T = len(I)
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El

        p_m_Vhalf = self.m_Vhalf
        p_m_k = self.m_k
        p_m_tau = self.m_tau

        p_h_Vhalf = self.h_Vhalf
        p_h_k = self.h_k
        p_h_tau = self.h_tau

        p_n_Vhalf = self.n_Vhalf
        p_n_k = self.n_k
        p_n_tau = self.n_tau

        p_E_K = self.E_K

        p_gbar_K1 = self.gbar_K1
        p_gbar_K2 = self.gbar_K2

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")

        m = np.zeros_like(V, dtype="double")
        h = np.zeros_like(V, dtype="double")
        n = np.zeros_like(V, dtype="double")

        # Set initial condition
        V[0] = V0

        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)

        code = """
                #include <math.h>
                #include <stdio.h>


                // DECLARE IMPORTED PARAMETERS

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);

                float m_Vhalf    = float(p_m_Vhalf);
                float m_k        = float(p_m_k);
                float m_tau      = float(p_m_tau);

                float h_Vhalf    = float(p_h_Vhalf);
                float h_k        = float(p_h_k);
                float h_tau      = float(p_h_tau);

                float n_Vhalf    = float(p_n_Vhalf);
                float n_k        = float(p_n_k);
                float n_tau      = float(p_n_tau);

                float E_K        = float(p_E_K);

                float gbar_K1    = float(p_gbar_K1);
                float gbar_K2    = float(p_gbar_K2);


                // DECLARE ADDITIONAL VARIABLES

                float m_inf_t;
                float h_inf_t;
                float n_inf_t;

                float DF_K_t;
                float gk_1_term;
                float gk_2_term;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = 1/(1 + exp(-m_k * (V[t-1] - m_Vhalf)));
                    m[t] = m[t-1] + dt/m_tau*(m_inf_t - m[t-1]);

                    // INTEGRATE h GATE
                    h_inf_t = 1/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = 1/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE K CONDUCTANCES
                    DF_K_t = V[t-1] - E_K;
                    gk_1_term = -DF_K_t * m[t] * h[t] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gk_1_term + gk_2_term);


                }

                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_tau',
                'p_h_Vhalf', 'p_h_k', 'p_h_tau',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau',
                'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V', 'I', 'm', 'h', 'n']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        return (time, V, m, h, n)

    def simulateDeterministic_forceSpikes(self, *args):
        """
        Subthreshold model does not spike.
        """

        raise RuntimeError('Subthreshold model does not spike.')

    def simulateVClamp(self, duration, V_const, V_pre, incl_gl=True, do_plot=False):
        """
        Compute the holding current elicited by a voltage step from V_pre to V_const
        """

        if V_pre is None:
            V_pre = V_const

        V_vec = np.ones(int(duration / self.dt), dtype=np.float64)
        V_vec *= V_const

        # Initialize vectors with equilibrium gating states.
        mInf_vec = self.mInf(V_vec)
        hInf_vec = self.hInf(V_vec)
        nInf_vec = self.nInf(V_vec)

        # Set initial condition to equilibrium state at V_pre.
        mInf_vec[0] = self.mInf(V_pre)
        hInf_vec[0] = self.hInf(V_pre)
        nInf_vec[0] = self.nInf(V_pre)

        # Compute gating state as a function of time.
        m_vec = self.computeGating(V_vec, mInf_vec, self.m_tau)
        h_vec = self.computeGating(V_vec, hInf_vec, self.h_tau)
        n_vec = self.computeGating(V_vec, nInf_vec, self.n_tau)

        # Compute active conductance vectors.
        gk1_vec = self.gbar_K1 * m_vec * h_vec
        gk2_vec = self.gbar_K2 * n_vec

        # Compute driving force vectors.
        DF_leak = V_vec - self.El
        DF_K = V_vec - self.E_K

        # Compute clamping current.
        if incl_gl:
            I_vec = self.gl * DF_leak + (gk1_vec + gk2_vec) * DF_K
        else:
            I_vec = 0. * DF_leak + (gk1_vec + gk2_vec) * DF_K

        # Optionally, plot the resulting current.
        if do_plot:
            NotImplemented

        return (V_vec, I_vec)

    ########################################################################################################
    # METHODS FOR MODEL FITTING
    ########################################################################################################

    def fit(self, experiment):
        """
        Fit the GIF model on experimental data.
        The experimental data are stored in the object experiment provided as an input.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit GIF"
        print "################################\n"

        self.fitSubthresholdDynamics(experiment)

    ########################################################################################################
    # FIT VOLTAGE RESET GIVEN ABSOLUTE REFRACOTORY PERIOD (step 1)
    ########################################################################################################

    def fitVoltageReset(self, experiment, Tref, do_plot=False):
        """
        Subthreshold model does not spike.
        """

        raise RuntimeError('Subthreshold model does not spike.')

    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################

    def fitSubthresholdDynamics(self, experiment):
        """
        Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        experiment: Experiment object on which the model is fitted.
        DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
        """

        print "\nGIF MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        self.dt = experiment.dt

        # Build X matrix and Y vector to perform linear regression (use all traces in training set)
        # For each training set an X matrix and a Y vector is built.
        ####################################################################################################
        X = []
        Y = []

        cnt = 0

        for tr in experiment.trainingset_traces:

            if tr.useTrace:

                cnt += 1
                reprint("Compute X matrix for repetition %d" % (cnt))

                # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)

                X.append(X_tmp)
                Y.append(Y_tmp)

        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        ####################################################################################################
        if cnt == 1:
            X = X[0]
            Y = Y[0]

        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        else:
            print "\nError, at least one training set trace should be selected to perform fit."

        # Perform linear Regression defined in Eq. 17 of Pozzorini et al. PLOS Comp. Biol. 2015
        ####################################################################################################

        print "\nPerform linear regression..."
        XTX = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY = np.dot(np.transpose(X), Y)
        b = np.dot(XTX_inv, XTY)
        b = b.flatten()

        # Extract explicit model parameters from regression result b
        ####################################################################################################

        self.C = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl

        self.gbar_K1 = b[3] * self.C
        self.gbar_K2 = b[4] * self.C

        self.printParameters()

        # Compute percentage of variance explained on dV/dt
        ####################################################################################################

        self.dV_data = Y.flatten()
        self.dV_fitted = np.dot(X, b).flatten()

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X, b))**2)/np.var(Y)

        self.var_explained_dV = var_explained_dV
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)

        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data

        self.I_data = []
        self.V_data = []
        self.V_sim = []
        self.m_sim = []
        self.h_sim = []
        self.n_sim = []

        for tr in experiment.trainingset_traces:

            if tr.useTrace:

                # Simulate subthreshold dynamics
                (time, V_est, m_est, h_est, n_est) = self.simulate(tr.I, tr.V[0])

                # Store data used for simulation along with simulated points
                self.I_data.append(tr.I)
                self.V_data.append(tr.V)
                self.V_sim.append(V_est)
                self.m_sim.append(m_est)
                self.h_sim.append(h_est)
                self.n_sim.append(n_est)

                # Compute SSE on points in ROI
                indices_tmp = tr.getROI()

                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        var_explained_V = 1.0 - SSE / VAR

        self.var_explained_V = var_explained_V
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace):
        """
        Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression
        defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
        The input parameter trace is an ojbect of class Trace.
        """

        # Select region where to perform linear regression (specified in the ROI of individual taces)
        ####################################################################################################
        selection = trace.getROI()
        selection_l = len(selection)

        # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
        ####################################################################################################
        X = np.zeros((selection_l, 5))

        # Compute equilibrium state of each gate
        m_inf_vec = self.mInf(trace.V)
        h_inf_vec = self.hInf(trace.V)
        n_inf_vec = self.nInf(trace.V)

        # Compute time-dependent state of each gate over whole trace
        m_vec = self.computeGating(trace.V, m_inf_vec, self.m_tau)
        h_vec = self.computeGating(trace.V, h_inf_vec, self.h_tau)
        n_vec = self.computeGating(trace.V, n_inf_vec, self.n_tau)

        # Compute gating state of each conductance over whole trace
        gating_vec_1 = m_vec * h_vec
        gating_vec_2 = n_vec

        # Compute K driving force over whole trace
        DF_K = self.getDF_K(trace.V)

        # Fill first three columns of X matrix
        X[:, 0] = trace.V[selection]
        X[:, 1] = trace.I[selection]
        X[:, 2] = np.ones(selection_l)

        # Fill K-conductance columns
        X[:, 3] = -(gating_vec_1 * DF_K)[selection]
        X[:, 4] = -(gating_vec_2 * DF_K)[selection]

        # Build Y vector (voltage derivative \dot_V_data)
        ####################################################################################################
        Y = (np.gradient(trace.V) / trace.dt)[selection]
        #Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]

        return (X, Y)

    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################

    def fitStaticThreshold(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    def fitThresholdDynamics(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    def maximizeLikelihood(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    def computeLikelihoodGradientHessian(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    def buildXmatrix_staticThreshold(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    def buildXmatrix_dynamicThreshold(self, *args):
        """
        Subthreshold models do not spike.
        """

        raise RuntimeError('Subthreshold models do not spike.')

    ########################################################################################################
    # METHODS FOR ASSESSING RESIDUALS
    ########################################################################################################

    def getResiduals_V(self, bins=None):
        """
        Bins can be None, an intger number of bins, or a vector with specifc points to use for binning.

        Returns a tupple of the form (values_V, residuals_V) for plotting.
        """

        residuals_V_tmp = []
        for i in range(len(self.V_data)):

            residuals_V_tmp.append(self.V_sim[i] - self.V_data[i])

        # Bin.
        if bins is None:
            values_V = [sw for sw in self.V_data]
        else:

            values_V = []
            residuals_V = []
            for i in range(len(self.V_data)):
                residuals_i, bin_edges, bin_no = binned_statistic(self.V_data[i],
                                                                residuals_V_tmp[i],
                                                                statistic='mean',
                                                                bins=bins)

                residuals_V.append(residuals_i)

                bin_centres = (bin_edges[1:] + bin_edges[:-1])/2.
                values_V.append(bin_centres)

        residuals_V = np.concatenate(residuals_V, axis=-1)
        values_V = np.concatenate(values_V, axis=-1)

        return values_V, residuals_V

    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################

    def plotFit(self, title=None):
        """
        Compare the real and simulated training sets.
        """

        plt.figure(figsize=(10, 5))

        V_p = plt.subplot(211)
        plt.title('Voltage traces')
        plt.ylabel('V (mV)')
        plt.xlabel('Time (ms)')

        dV_p = plt.subplot(212)
        plt.title('dV traces')
        plt.ylabel('dV/dt (mV/ms)')
        plt.xlabel('Time (ms)')

        t_V = np.arange(0, int(np.round(len(self.V_data[0])*self.dt)), self.dt)
        t_dV = np.arange(0, int(np.round(len(self.dV_data)*self.dt)), self.dt)

        assert len(t_V) == len(self.V_data[0]), 'time and V_vectors not of equal lengths'
        assert len(t_dV) == len(self.dV_data), 'time and dV_vectors not of equal lengths'

        for i in range(len(self.V_data)):

            # Only label the first line.
            if i == 0:
                V_p.plot(t_V, self.V_data[i], 'k-', linewidth=0.5, label='Real')
                V_p.plot(t_V, self.V_sim[i], 'r-', linewidth=0.5, alpha=0.7, label='Simulated')

            else:
                V_p.plot(t_V, self.V_data[i], 'k-', linewidth=0.5)
                V_p.plot(t_V, self.V_sim[i], 'r-', linewidth=0.5)

        dV_p.plot(t_dV, self.dV_data, 'k-', label='Real')
        dV_p.plot(t_dV, self.dV_fitted, 'r-', alpha=0.7, label='Fitted')

        V_p.legend()
        dV_p.legend()

        plt.tight_layout()

        if title is not None:
            plt.suptitle(title)
            plt.subplots_adjust(top=0.85)

        plt.show()

    def plotPowerSpectrumDensity(self, title=None):
        """
        Compare power spectrum densities of model and real neuron in training data.

        Only uses first training sweep.
        """

        GIF_f, GIF_PSD, _, _ = Trace(self.V_sim[0],
                               self.I_data[0],
                               len(self.V_sim[0])*self.dt,
                               self.dt).extractPowerSpectrumDensity()

        Data_f, Data_PSD, _, _ = Trace(self.V_data[0],
                                 self.I_data[0],
                                 len(self.V_data[0])*self.dt,
                                 self.dt).extractPowerSpectrumDensity()

        _, _, f_I, PSD_I = Trace(self.V_sim[0],
                               self.I_data[0],
                               len(self.V_sim[0])*self.dt,
                               self.dt).extractPowerSpectrumDensity()

        plt.figure(figsize=(10, 4))

        ax = plt.subplot(211)
        ax.set_xscale('log')

        ax.plot(Data_f, Data_PSD, 'k-', linewidth=0.5, label='Real')
        ax.plot(GIF_f, GIF_PSD, 'r-', linewidth=0.5, label='Simulated')
        ax.plot(f_I, PSD_I, 'b-', linewidth=0.5, label='Input')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.legend()

        ax2 = plt.subplot(212, sharex=ax)
        ax2.set_xscale('log')

        ax2.plot(Data_f, Data_PSD / PSD_I, 'k-', linewidth=0.5, label='Real (norm.)')
        ax2.plot(GIF_f, GIF_PSD / PSD_I, 'r-', linewidth=0.5, label='Simulated (norm.)')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (norm.)')
        ax2.legend()

        plt.tight_layout()

        if title is not None:
            plt.suptitle(title)
            plt.subplots_adjust(top=0.85)

        plt.show()

    def plotGating(self):

        plt.figure()
        g_p = plt.subplot(111)
        plt.title('Simulated gating parameters')
        plt.ylabel('g')
        plt.xlabel('Time (ms)')

        t = np.arange(0, int(np.round(len(self.m_sim[0])*self.dt)), self.dt)

        assert len(t) == len(self.m_sim[0]), 'time and simulated gating vectors not of equal lengths'

        for i in range(len(self.m_sim)):

            # Only label the first line.
            if i == 0:

                g_p.plot(t, self.m_sim[i], label='m')
                g_p.plot(t, self.h_sim[i], label='h')
                g_p.plot(t, self.n_sim[i], label='n')

            else:

                g_p.plot(t, self.m_sim[i])
                g_p.plot(t, self.h_sim[i])
                g_p.plot(t, self.n_sim[i])

        g_p.legend()

        plt.tight_layout()
        plt.show()

    def plotParameters(self):
        """
        Generate figure with model filters.
        """

        plt.figure(facecolor='white', figsize=(5, 4))

        # Plot kappa
        plt.subplot(1, 1, 1)

        K_support = np.linspace(0, 150.0, 300)
        K = 1./self.C*np.exp(-K_support/(self.C/self.gl))

        plt.plot(K_support, K, color='red', lw=2)
        plt.plot([K_support[0], K_support[-1]], [0, 0], ls=':', color='black', lw=2)

        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane filter (MOhm/ms)")

        plt.tight_layout()

        plt.show()

    def printParameters(self):
        """
        Print model parameters on terminal.
        """

        print "\n-------------------------"
        print "GIF model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f" % (self.C/self.gl)
        print "R (MOhm):\t%0.3f" % (1.0/self.gl)
        print "C (nF):\t\t%0.3f" % (self.C)
        print "gl (nS):\t%0.6f" % (self.gl)
        print "El (mV):\t%0.3f" % (self.El)
        print "gbar_K1:\t%0.6f" % (self.gbar_K1)
        print "gbar_K2:\t%0.6f" % (self.gbar_K2)
        print "-------------------------\n"

    @classmethod
    def compareModels(cls, GIFs, labels=None):
        """
        Given a list of GIF models, GIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS

        print "\n#####################################"
        print "GIF model comparison"
        print "#####################################\n"

        cnt = 0
        for GIF in GIFs:

            #print "Model: " + labels[cnt]
            GIF.printParameters()
            cnt += 1

        print "#####################################\n"

        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9, 8))

        colors = plt.cm.jet(np.linspace(0.7, 1.0, len(GIFs)))

        # Membrane filter
        plt.subplot(111)

        cnt = 0
        for GIF in GIFs:

            K_support = np.linspace(0, 150.0, 1500)
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1

        plt.plot([K_support[0], K_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)

        plt.show()

    @classmethod
    def plotAverageModel(cls, GIFs):
        """
        Average model parameters and plot summary data.
        """

        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################

        fig = plt.figure(facecolor='white', figsize=(16, 7))
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.90, wspace=0.35, hspace=0.5)
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'

        # MEMBRANE FILTER
        #######################################################################################################

        plt.subplot(2, 4, 1)

        K_all = []

        for GIF in GIFs:

            K_support = np.linspace(0, 150.0, 300)
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)

            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std = np.std(K_all, axis=0)

        plt.fill_between(K_support, K_mean+K_std, y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)
        plt.plot([K_support[0], K_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')

        # R
        #######################################################################################################

        plt.subplot(4, 6, 12+1)

        p_all = []
        for GIF in GIFs:

            p = 1./GIF.gl
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

        # tau_m
        #######################################################################################################

        plt.subplot(4, 6, 18+1)

        p_all = []
        for GIF in GIFs:

            p = GIF.C/GIF.gl
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

        # El
        #######################################################################################################

        plt.subplot(4, 6, 12+2)

        p_all = []
        for GIF in GIFs:

            p = GIF.El
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])