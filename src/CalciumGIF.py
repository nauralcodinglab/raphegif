#%% IMPORT MODULES

from __future__ import division

import weave
import numpy as np
import numba as nb

from src.AugmentedGIF import AugmentedGIF
import src.Tools as Tools


#%% DEFINE CALCIUM GIF CLASS

class CalciumGIF(AugmentedGIF):

    def __init__(self, dt=0.1):

        super(CalciumGIF, self).__init__(dt=dt)

        # Parameters related to potassium conductances
        self.m_Vhalf = -60.2
        self.m_k = 0.0446
        self.m_A = 1.

        self.h_Vhalf = -63.
        self.h_k = -0.123
        self.h_A = 1.
        del self.h_tau

        self.E_Ca = 60

        self.n_Vhalf = -24.3
        self.n_k = 0.216
        self.n_A = 1.55
        self.n_tau = 1

        self.E_K = -101

        self.gbar_K1 = 0.010
        self.gbar_K2 = 0.001


    ### Calcium gate related methods.

    def mInf(self, V):

        """Compute the equilibrium activation gate state of the calcium conductance.
        """

        return (self.m_A/(1 + np.exp(-self.m_k * (V - self.m_Vhalf))))**4


    def hInf(self, V):

        """Compute the equilibrium state of the inactivation gate of the calcium conductance.
        """

        return self.h_A/(1 + np.exp(-self.h_k * (V - self.h_Vhalf)))

    @staticmethod
    @nb.jit(cache = True, nopython = True)
    def m_tau(V):

        """Compute the activation time of the calcium conductance.
        """

        return 2.4 + 22.5/np.cosh((V + 39.)/12.)

    @staticmethod
    @nb.jit(cache = True, nopython = True)
    def h_tau(V):

        """Compute the inactivation time of the calcium conductance.
        """

        return 127. + 0.21 * np.exp(np.minimum(50, -V)/6.5)


    def computeGating(self, V, inf_vec, tau):

        """
        Compute the state of a gate over time.

        Wrapper for _computeGatingInternal, which is a nb.jit-accelerated static method.

        tau can be a function or float.
        """

        try:
            tau_vec = tau(V)
        except TypeError:
            tau_vec = np.array([tau] * len(V), dtype = np.float64)

        return self._computeGatingInternal(V, inf_vec, tau_vec, self.dt)


    @staticmethod
    @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64))
    def _computeGatingInternal(V, inf_vec, tau_vec, dt):

        """
        Internal method called by computeGating.
        """

        output = np.empty_like(V, dtype = np.float64)
        output[0] = inf_vec[0]

        for i in range(1, len(V)):

            output[i] = output[i - 1] + (inf_vec[i - 1] - output[i - 1])/tau_vec[i - 1] * dt

        return output

    def getDF_Ca(self, V):
        return V - self.E_Ca


    ### Simulation methods.

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
        p_T         = len(I)
        p_dt        = self.dt

        # Model parameters
        p_gl        = self.gl
        p_C         = self.C
        p_El        = self.El

        p_m_Vhalf   = self.m_Vhalf
        p_m_k       = self.m_k
        p_m_A       = self.m_A

        p_h_Vhalf   = self.h_Vhalf
        p_h_k       = self.h_k
        p_h_A       = self.h_A

        p_E_Ca      = self.E_Ca

        p_n_Vhalf   = self.n_Vhalf
        p_n_k       = self.n_k
        p_n_A       = self.n_A
        p_n_tau     = self.n_tau

        p_E_K       = self.E_K

        p_gbar_K1   = self.gbar_K1
        p_gbar_K2   = self.gbar_K2

        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Vt_star   = self.Vt_star
        p_DV        = self.DV
        p_lambda0   = self.lambda0

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma     = p_gamma.astype('double')
        p_gamma_l   = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

        m = np.zeros_like(V, dtype = "double")
        h = np.zeros_like(V, dtype = "double")
        n = np.zeros_like(V, dtype = "double")

        # Set initial condition
        V[0] = V0

        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)


        code =  """
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

                float E_Ca       = float(p_E_Ca);

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

                float m_tau;
                float h_tau;

                float DF_Ca_t;
                float DF_K_t;

                float gCa_term;
                float gk_2_term;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = pow(m_A/(1 + exp(-m_k * (V[t-1] - m_Vhalf))), 4);
                    m_tau = 2.4 + 22.5/cosh((V[t-1] + 39.)/12.);
                    m[t] = m[t-1] + dt/m_tau*(m_inf_t - m[t-1]);

                    // INTEGRATE h GATE
                    h_inf_t = h_A/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    if (V[t-1] <= -50.) {
                        h_tau = 127. + 0.21 * exp(50./6.5);
                    } else {
                        h_tau = 127. + 0.21 * exp(-V[t-1]/6.5);
                    };
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = n_A/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE CONDUCTANCES
                    DF_Ca_t = V[t-1] - E_Ca;
                    DF_K_t = V[t-1] - E_K;
                    gCa_term = -DF_Ca_t * m[t-1] * h[t-1] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t-1] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gCa_term + gk_2_term - eta_sum[t-1]);


                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t-1]-Vt_star-gamma_sum[t-1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    r = rand()/rand_max;
                    if (r > p_dontspike) {

                        spks[t+1] = 1.0;

                        t = t + Tref_ind;

                        V[t] = Vr;


                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+j] += p_gamma[j];

                    }

                }

                """

        vars = [ 'p_T','p_dt','p_gl','p_C','p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_A',
                'p_h_Vhalf', 'p_h_k', 'p_h_A',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau', 'p_n_A',
                'p_E_Ca', 'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V','I','m','h','n',
                'p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0',
                'p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        eta_sum   = eta_sum[:p_T]
        V_T = gamma_sum[:p_T] + p_Vt_star

        spks = (np.where(spks==1)[0])*self.dt

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
        p_T         = len(I)
        p_dt        = self.dt

        # Model parameters
        p_gl        = self.gl
        p_C         = self.C
        p_El        = self.El

        p_m_Vhalf   = self.m_Vhalf
        p_m_k       = self.m_k
        p_m_A       = self.m_A

        p_h_Vhalf   = self.h_Vhalf
        p_h_k       = self.h_k
        p_h_A       = self.h_A

        p_E_Ca      = self.E_Ca

        p_n_Vhalf   = self.n_Vhalf
        p_n_k       = self.n_k
        p_n_A       = self.n_A
        p_n_tau     = self.n_tau

        p_E_K       = self.E_K

        p_gbar_K1   = self.gbar_K1
        p_gbar_K2   = self.gbar_K2

        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Tref_i    = int(self.Tref/self.dt)

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(spks, dtype="double")
        spks_i   = Tools.timeToIndex(spks, self.dt)

        m = np.zeros_like(V, dtype = "double")
        h = np.zeros_like(V, dtype = "double")
        n = np.zeros_like(V, dtype = "double")

        # Compute adaptation current (sum of eta triggered at spike times in spks)
        eta_sum  = np.array(
                np.zeros(p_T + int(1.1*p_eta_l) + p_Tref_i),
                dtype="double")

        for s in spks_i :
            eta_sum[s + 1 + p_Tref_i  : s + 1 + p_Tref_i + p_eta_l] += p_eta

        eta_sum  = eta_sum[:p_T]

        # Set initial condition
        V[0] = V0

        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)


        code =  """
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

                float E_Ca       = float(p_E_Ca);

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

                float m_tau;
                float h_tau;

                float DF_Ca_t;
                float DF_K_t;

                float gCa_term;
                float gk_2_term;

                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;

                for (int t=1; t<T_ind; t++) {

                    // INTEGRATE m GATE
                    m_inf_t = pow(m_A/(1 + exp(-m_k * (V[t-1] - m_Vhalf))), 4);
                    m_tau = 2.4 + 22.5/cosh((V[t-1] + 39.)/12.);
                    m[t] = m[t-1] + dt/m_tau*(m_inf_t - m[t-1]);

                    // INTEGRATE h GATE
                    h_inf_t = h_A/(1 + exp(-h_k * (V[t-1] - h_Vhalf)));
                    if (V[t-1] <= -50.) {
                        h_tau = 127. + 0.21 * exp(50./6.5);
                    } else {
                        h_tau = 127. + 0.21 * exp(-V[t-1]/6.5);
                    };
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);

                    // INTEGRATE n GATE
                    n_inf_t = n_A/(1 + exp(-n_k * (V[t-1] - n_Vhalf)));
                    n[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);

                    // COMPUTE CONDUCTANCES
                    DF_Ca_t = V[t-1] - E_Ca;
                    DF_K_t = V[t-1] - E_K;
                    gCa_term = -DF_Ca_t * m[t-1] * h[t-1] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t-1] * gbar_K2;

                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gCa_term + gk_2_term - eta_sum[t-1]);

                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;
                        V[t] = Vr ;
                    }

                }

                """

        vars = [ 'p_T','p_dt','p_gl','p_C','p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_A',
                'p_h_Vhalf', 'p_h_k', 'p_h_A',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau', 'p_n_A',
                'p_E_Ca', 'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V','I','m','h','n',
                'p_Vr','p_Tref',
                'eta_sum','spks', 'spks_i']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum   = eta_sum[:p_T]

        return (time, V, eta_sum)

#%% SIMPLE TESTS

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    test_CaGIF = CalciumGIF(0.1)

    # Check equilibrium stuff.
    V_vec = np.arange(-100, -10, 0.1)
    plt.figure()
    plt.subplot(121)
    plt.title('Equilibrium gating')
    plt.plot(V_vec, test_CaGIF.mInf(V_vec), 'k-', label = 'm')
    plt.plot(V_vec, test_CaGIF.hInf(V_vec), 'r-', label = 'h')
    plt.xlabel('V')
    plt.legend()

    plt.subplot(122)
    plt.title('Kinetics')
    plt.plot(V_vec, test_CaGIF.m_tau(V_vec), 'k-', label = 'm')
    plt.plot(V_vec, test_CaGIF.h_tau(V_vec), 'r-', label = 'h')
    plt.ylabel('tau (ms)')
    plt.xlabel('V (mV)')
    plt.legend()

    plt.tight_layout()

    plt.show()

    V_step = np.concatenate([-70 * np.ones(1000), -40 * np.ones(10000)])
    plt.figure()
    plt.subplot(111)
    plt.plot(test_CaGIF.computeGating(V_step, test_CaGIF.mInf(V_step), test_CaGIF.m_tau), 'k-', label = 'm')
    plt.plot(test_CaGIF.computeGating(V_step, test_CaGIF.hInf(V_step), test_CaGIF.h_tau), 'r-', label = 'h')
    plt.plot(test_CaGIF.computeGating(V_step, test_CaGIF.nInf(V_step), test_CaGIF.n_tau), label = 'n')
    plt.legend()
    plt.show()

    # Test simulate method.
    print 'Testing CalciumGIF.simulate method.'
    plt.figure()
    plt.subplot(111)
    plt.plot(test_CaGIF.simulate(np.concatenate([-0.03 * np.ones(5000), 0.25 * np.ones(10000)]), -70)[1])
    plt.show()
    print 'CalciumGIF.simulate method runs!'

    # Test forced spike simulation method.
    print 'Testing CalciumGIF.simulateDeterministic_forceSpikes method'
    plt.figure()
    plt.subplot(111)
    plt.plot(test_CaGIF.simulateDeterministic_forceSpikes(
        np.concatenate([-0.03 * np.ones(5000), 0.25 * np.ones(10000)]), -70, [1000]
        )[1])
    plt.show()
    print 'CalciumGIF.simulateDeterministic_forceSpikes runs!'
