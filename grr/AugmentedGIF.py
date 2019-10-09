import math
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import weave
from scipy import optimize

from .GIF import GIF
from .Filter_Rect import Filter_Rect_LogSpaced
from . import Tools
from .Tools import reprint


#%% DEFINE AUGMENTED GIF CLASS

class AugmentedGIF(GIF):

    def __init__(self, dt=0.1):

        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)

        # Define model parameters

        self.gl = 0.001        # nS, leak conductance
        self.C = 0.1     # nF, capacitance
        self.El = -65.0            # mV, reversal potential

        self.Vr = -50.0            # mV, voltage reset
        self.Tref = 4.0              # ms, absolute refractory period

        self.Vt_star = -48.0            # mV, steady state voltage threshold VT*
        self.DV = 0.5              # mV, threshold sharpness
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz

        self.eta = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
        self.gamma = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)

        # Initialize the spike-triggered current eta with an exponential function

        def expfunction_eta(x):
            return 0.2*np.exp(-x/100.0)

        self.eta.setFilter_Function(expfunction_eta)

        # Initialize the spike-triggered current gamma with an exponential function

        def expfunction_gamma(x):
            return 0.01*np.exp(-x/100.0)

        self.gamma.setFilter_Function(expfunction_gamma)

        # Variables related to fitting procedure

        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0
        self.var_explained_dV = 0
        self.var_explained_V = 0

        # Parameters related to potassium conductances
        self.m_Vhalf = -23.7
        self.m_k = 0.0985
        self.m_A = 1.61

        self.h_Vhalf = -59.2
        self.h_k = -0.165
        self.h_A = 1.03
        self.h_tau = 70

        self.n_Vhalf = -24.3
        self.n_k = 0.216
        self.n_A = 1.55
        self.n_tau = 1

        self.E_K = -101

        self.gbar_K1 = 0.010
        self.gbar_K2 = 0.001

    def mInf(self, V):
        """Compute the equilibrium activation gate state of the potassium conductance.
        """

        return self.m_A/(1 + np.exp(-self.m_k * (V - self.m_Vhalf)))

    def hInf(self, V):
        """Compute the equilibrium state of the inactivation gate of the potassium conductance.
        """

        return self.h_A/(1 + np.exp(-self.h_k * (V - self.h_Vhalf)))

    def nInf(self, V):
        """Compute the equilibrium state of the non-inactivating conductance.
        """

        return self.n_A/(1 + np.exp(-self.n_k * (V - self.n_Vhalf)))

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

    def simulate(self, I, V0):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times
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
        p_m_A = self.m_A

        p_h_Vhalf = self.h_Vhalf
        p_h_k = self.h_k
        p_h_A = self.h_A
        p_h_tau = self.h_tau

        p_n_Vhalf = self.n_Vhalf
        p_n_k = self.n_k
        p_n_A = self.n_A
        p_n_tau = self.n_tau

        p_E_K = self.E_K

        p_gbar_K1 = self.gbar_K1
        p_gbar_K2 = self.gbar_K2

        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Vt_star = self.Vt_star
        p_DV = self.DV
        p_lambda0 = self.lambda0

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma = p_gamma.astype('double')
        p_gamma_l = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

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

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);

                float m_Vhalf    = float(p_m_Vhalf);
                float m_k        = float(p_m_k);
                float m_A        = float(p_m_A);

                float h_Vhalf    = float(p_h_Vhalf);
                float h_k        = float(p_h_k);
                float h_A        = float(p_h_A);
                float h_tau      = float(p_h_tau);

                float n_Vhalf    = float(p_n_Vhalf);
                float n_k        = float(p_n_k);
                float n_A        = float(p_n_A);
                float n_tau      = float(p_n_tau);

                float E_K        = float(p_E_K);

                float gbar_K1    = float(p_gbar_K1);
                float gbar_K2    = float(p_gbar_K2);

                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float Vt_star    = float(p_Vt_star);
                float DeltaV     = float(p_DV);
                float lambda0    = float(p_lambda0);

                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);


                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float r = 0.0;


                // DECLARE ADDITIONAL VARIABLES

                float m_inf_t;
                float h_inf_t;
                float n_inf_t;

                float DF_K_t;
                float gk_1_term;
                float gk_2_term;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = m_A/(1 + exp(-m_k * (V[t-1] - m_Vhalf)));
                    m[t] = m_inf_t;

                    // INTEGRATE h GATE
                    h_inf_t = h_A/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = n_A/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE K CONDUCTANCES
                    DF_K_t = V[t-1] - E_K;
                    gk_1_term = -DF_K_t * m[t-1] * h[t-1] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t-1] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gk_1_term + gk_2_term - eta_sum[t-1]);


                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t-1]-Vt_star-gamma_sum[t-1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    r = rand()/rand_max;
                    if (r > p_dontspike) {

                        spks[t] = 1.0;

                        t = t + Tref_ind;
                        if (t < T_ind)
                            V[t] = Vr;


                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+j] += p_gamma[j];

                    }

                }

                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_A',
                'p_h_Vhalf', 'p_h_k', 'p_h_tau', 'p_h_A',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau', 'p_n_A',
                'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V', 'I', 'm', 'h', 'n',
                'p_Vr', 'p_Tref', 'p_Vt_star', 'p_DV', 'p_lambda0',
                'p_eta', 'p_eta_l', 'eta_sum', 'p_gamma', 'gamma_sum', 'p_gamma_l', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        eta_sum = eta_sum[:p_T]
        V_T = gamma_sum[:p_T] + p_Vt_star

        spks = (np.where(spks == 1)[0])*self.dt

        return (time, V, eta_sum, V_T, spks)

    def simulateDeterministic_forceSpikes(self, I, V0, spks):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times
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
        p_m_A = self.m_A

        p_h_Vhalf = self.h_Vhalf
        p_h_k = self.h_k
        p_h_A = self.h_A
        p_h_tau = self.h_tau

        p_n_Vhalf = self.n_Vhalf
        p_n_k = self.n_k
        p_n_A = self.n_A
        p_n_tau = self.n_tau

        p_E_K = self.E_K

        p_gbar_K1 = self.gbar_K1
        p_gbar_K2 = self.gbar_K2

        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Tref_i = int(self.Tref/self.dt)

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(spks, dtype="double")
        spks_i = Tools.timeToIndex(spks, self.dt)

        m = np.zeros_like(V, dtype="double")
        h = np.zeros_like(V, dtype="double")
        n = np.zeros_like(V, dtype="double")

        # Compute adaptation current (sum of eta triggered at spike times in spks)
        eta_sum = np.array(
                np.zeros(p_T + int(1.1*p_eta_l) + p_Tref_i),
                dtype="double")

        for s in spks_i:
            eta_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_eta_l] += p_eta

        eta_sum = eta_sum[:p_T]

        # Set initial condition
        V[0] = V0

        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)

        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);

                float m_Vhalf    = float(p_m_Vhalf);
                float m_k        = float(p_m_k);
                float m_A        = float(p_m_A);

                float h_Vhalf    = float(p_h_Vhalf);
                float h_k        = float(p_h_k);
                float h_A        = float(p_h_A);
                float h_tau      = float(p_h_tau);

                float n_Vhalf    = float(p_n_Vhalf);
                float n_k        = float(p_n_k);
                float n_A        = float(p_n_A);
                float n_tau      = float(p_n_tau);

                float E_K        = float(p_E_K);

                float gbar_K1    = float(p_gbar_K1);
                float gbar_K2    = float(p_gbar_K2);

                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);


                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float r = 0.0;

                // DECLARE ADDITIONAL VARIABLES

                float m_inf_t;
                float h_inf_t;
                float n_inf_t;

                float DF_K_t;
                float gk_1_term;
                float gk_2_term;

                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = m_A/(1 + exp(-m_k * (V[t-1] - m_Vhalf)));
                    m[t] = m_inf_t;

                    // INTEGRATE h GATE
                    h_inf_t = h_A/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = n_A/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE K CONDUCTANCES
                    DF_K_t = V[t-1] - E_K;
                    gk_1_term = -DF_K_t * m[t-1] * h[t-1] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t-1] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gk_1_term + gk_2_term - eta_sum[t-1]);

                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;
                        V[t] = Vr ;
                    }

                }

                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_A',
                'p_h_Vhalf', 'p_h_k', 'p_h_tau', 'p_h_A',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau', 'p_n_A',
                'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V', 'I', 'm', 'h', 'n',
                'p_Vr', 'p_Tref',
                'eta_sum', 'spks', 'spks_i']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]

        return (time, V, eta_sum)

    ### Fitting related methods

    def fit(self, experiment, h_tau_candidates, DT_beforeSpike=5.0):
        """Fit the AugmentedGIF model to experimental data.

        Fit model to training data.

        Parameters
        ----------
        experiment : grr.Experiment object
            Experimental data to fit the model to.
        h_tau_candidates : list-like
            Values for inactivation time constant line search. The inactivation
            time constant of gK1 is treated as a hyperparameter. Following a
            line search, the h_tau and other parameters with the highest
            subthreshold variance explained are stored in the model.
        DT_beforeSpike : float
            Region of trainingset_traces before each spike to be ignored during
            fitting in ms.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit GIF"
        print "################################\n"

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        self.fitSubthresholdDynamics(
            experiment, h_tau_candidates, DT_beforeSpike=DT_beforeSpike
        )
        self.fitStaticThreshold(experiment)
        self.fitThresholdDynamics(experiment)

    def fitSubthresholdDynamics(self, experiment, h_tau_candidates, DT_beforeSpike=5.0):
        """
        Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        experiment: Experiment object on which the model is fitted.
        DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
        """

        print "\nAugmentedGIF MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        if not np.isclose(self.dt, experiment.dt):
            warnings.warn(
                'AugmentedGIF.dt ({moddt}ms) and experiment.dt ({exptdt}ms) '
                'do not match. Overwriting AugmentedGIF.dt.'
            )
            self.dt = experiment.dt

        # RUN REGRESSION WITH LINE SEARCH OVER H_TAU

        """Implementation note:
        Line search of possible h_tau values requires modifying instance h_tau
        attribute. Guarantee that h_tau is restored to its original value even
        if errors are thrown by encapsulating line search in a try/finally
        block.
        """
        old_h_tau = self.h_tau  # Save original h_tau.
        coeffs = []
        var_explained = []
        try:
            for h_tau_candidate in h_tau_candidates:
                self.h_tau = h_tau_candidate

                # Build regression matrices for each trace.
                X = []
                Y = []

                cnt = 0
                for tr in experiment.trainingset_traces:
                    if tr.useTrace:
                        cnt += 1
                        reprint("Compute X matrix for repetition %d" % (cnt))

                        # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
                        (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, DT_beforeSpike=DT_beforeSpike)
                        X.append(X_tmp)
                        Y.append(Y_tmp)

                # Concatenate matrices from different traces.
                if cnt == 1:
                    X = X[0]
                    Y = Y[0]
                elif cnt > 1:
                    X = np.concatenate(X, axis=0)
                    Y = np.concatenate(Y, axis=0)
                else:
                    raise RuntimeError(
                        'Trace.useTrace must be True for at least one Trace in '
                        'experiment.trainingset_traces in order to fit the model.'
                    )

                # Run bounded linear regression.
                betas = optimize.lsq_linear(
                    X, Y,
                    bounds = (
                        np.concatenate([
                            [-np.inf, 0, -np.inf, 0, 0],
                            np.full(X.shape[1] - 5, -np.inf)
                        ]),
                        np.concatenate([
                            [0, np.inf, np.inf, np.inf, np.inf],
                            np.full(X.shape[1] - 5, np.inf)
                        ])
                    )
                )['x'].tolist()
                coeffs.append(betas)

                # Compute variance explained.
                Yhat = np.dot(X, betas)
                MSE = np.mean((Y - Yhat)**2)
                var_explained.append(1. - MSE/np.var(Y))

        finally:
            self.h_tau = old_h_tau

        # Assign model parameters based on best regression result.
        best_coeffs = np.array(coeffs[np.argmax(var_explained)])

        self.C = 1./best_coeffs[1]
        self.gl = -best_coeffs[0] * self.C
        self.El = best_coeffs[2] * self.C / self.gl

        self.gbar_K1 = best_coeffs[3] * self.C
        self.gbar_K2 = best_coeffs[4] * self.C
        self.h_tau = h_tau_candidates[np.argmax(var_explained)]

        self.eta.setFilter_Coefficients(-best_coeffs[5:] * self.C)

        self.hyperparameter_search = {
            'h_tau': h_tau_candidates,
            'var_explained': var_explained
        }

        self.printParameters()

        self.var_explained_dV = np.max(var_explained)
        print "Percentage of variance explained (on dV/dt): %0.2f" % (self.var_explained_dV*100.0)

        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data
        for tr in experiment.trainingset_traces:
            if tr.useTrace:
                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)
                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        var_explained_V = 1.0 - SSE / VAR

        self.var_explained_V = var_explained_V
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, DT_beforeSpike=5.0):
        """
        Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression
        defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
        The input parameter trace is an ojbect of class Trace.
        """

        # Length of the voltage trace
        Tref_ind = int(self.Tref/trace.dt)

        # Select region where to perform linear regression (specified in the ROI of individual taces)
        ####################################################################################################
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)

        # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
        ####################################################################################################
        X = np.zeros((selection_l, 5))

        # Compute equilibrium state of each gate
        m_inf_vec = self.mInf(trace.V)
        h_inf_vec = self.hInf(trace.V)
        n_inf_vec = self.nInf(trace.V)

        # Compute time-dependent state of each gate over whole trace
        m_vec = m_inf_vec
        h_vec = self.computeGating(trace.V, h_inf_vec, self.h_tau)
        n_vec = self.computeGating(trace.V, n_inf_vec, self.n_tau)

        # Compute gating state of each conductance over whole trace
        gating_vec_1 = m_vec * h_vec
        gating_vec_2 = n_vec

        # Compute K driving force over whole trace
        DF_K = self.getDF_K(trace.V)

        # Fill first two columns of X matrix
        X[:, 0] = trace.V[selection]
        X[:, 1] = trace.I[selection]
        X[:, 2] = np.ones(selection_l)

        # Fill K-conductance columns
        X[:, 3] = -(gating_vec_1 * DF_K)[selection]
        X[:, 4] = -(gating_vec_2 * DF_K)[selection]

        # Compute and fill the remaining columns associated with the spike-triggered current eta
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt)
        X = np.concatenate((X, X_eta[selection, :]), axis=1)

        # Build Y vector (voltage derivative \dot_V_data)
        ####################################################################################################
        Y = (np.gradient(trace.V)/trace.dt)[selection]

        return (X, Y)

    def printParameters(self):
        """
        Print model parameters on terminal.
        """

        print "\n-------------------------"
        print "AugmentedGIF model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f" % (self.C/self.gl)
        print "R (MOhm):\t%0.3f" % (1.0/self.gl)
        print "C (nF):\t\t%0.3f" % (self.C)
        print "gl (nS):\t%0.6f" % (self.gl)
        print "El (mV):\t%0.3f" % (self.El)
        print "Tref (ms):\t%0.3f" % (self.Tref)
        print "Vr (mV):\t%0.3f" % (self.Vr)
        print "Vt* (mV):\t%0.3f" % (self.Vt_star)
        print "DV (mV):\t%0.3f" % (self.DV)
        print "gk1 (ns):\t%0.4f" % (self.gbar_K1)
        print "gk2 (ns):\t%0.4f" % (self.gbar_K2)
        print "-------------------------\n"
